"""ScoreToF0 (S58) — the SEPARATE generative F0 model (DAW auto-pitch toggle).

Score → small content-aware encoder → x1-prediction FM that GENERATES cents-deviation-from-note
(the one-to-many expression — vibrato/scoops/transitions — that a deterministic head AVERAGES into a
flat, 跑调 pitch). 1-D target ⇒ none of the 768-D cv-FM failure modes apply (no thin manifold). Plus a
deterministic voicing head (SP-relaxed: GT voicing, not forced-0 at SP). Fully decoupled from the cv
model (own encoder reads phonemes ⇒ content-aware F0 microprosody). Deploy: f0 = note·2^(cents/1200).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConformerEncoder
from .score2contentvec import DiTBlock, LengthRegulator   # reuse the proven FM block + soft upsampler


def _fill_voicing_gaps(voiced, k):
    """Fill UNvoiced runs <= k frames that are SANDWICHED between voiced (the voicing head briefly
    over-unvoices at consonants/SP; restoring them = matching GT, pitch glides through naturally)."""
    v = voiced.copy(); T = len(v); i = 0
    while i < T:
        if not v[i]:
            j = i
            while j < T and not v[j]:
                j += 1
            if (j - i) <= k and i > 0 and j < T:    # runs alternate ⇒ i>0,j<T means both edges voiced
                v[i:j] = True
            i = j
        else:
            i += 1
    return v


class F0FlowDecoder(nn.Module):
    """x1-prediction FM over the 1-D cents-deviation curve, conditioned on the score (frame_hidden).
    Mirrors the cv FlowMatchingDecoder but in/out are 1-channel. DiT-FiLM blocks (tanh-bounded = stable)."""

    def __init__(self, cond_dim, dim=256, n_blocks=4, n_heads=4, conv_kernel=7):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Conv1d(1, dim, 1)
        self.cond_in = nn.Linear(cond_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads, conv_kernel) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_proj = nn.Conv1d(dim, 1, 1)

    def _time_emb(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * fr[None]
        return self.time_mlp(torch.cat([a.sin(), a.cos()], -1))

    def _pos_emb(self, T, dev):
        half = self.dim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=dev) / half)
        a = torch.arange(T, device=dev)[:, None] * fr[None]
        return torch.cat([a.sin(), a.cos()], -1).unsqueeze(0)

    def forward(self, x_t, t, c, mask=None):
        """x_t [B,1,T]; t [B]; c [B,T,cond_dim]. Returns x1_hat [B,1,T]."""
        B, _, T = x_t.shape
        cond = self.cond_in(c) + self._pos_emb(T, x_t.device)
        t_emb = self._time_emb(t)
        h = self.in_proj(x_t).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h, cond, t_emb, mask)
        return self.out_proj(self.out_norm(h).transpose(1, 2))


class ScoreToF0(nn.Module):
    SP_ID = 3

    def __init__(self, config):
        super().__init__()
        pc = config["phoneme"]; sc = config["speaker"]; ec = config["encoder"]; lrc = config["length_regulator"]
        lc = config.get("language", {}); lang_dim = lc.get("embed_dim", 64)
        self.phone_embed = nn.Embedding(pc["vocab_size"], pc["embed_dim"], padding_idx=pc["padding_idx"])
        self.pitch_embed = nn.Embedding(128, pc["embed_dim"])
        self.speaker_embed = nn.Embedding(sc["n_speakers"], sc["embed_dim"])
        self.language_embed = nn.Embedding(lc.get("n_langs", 9), lang_dim)
        self.input_proj = nn.Linear(pc["embed_dim"] * 2 + sc["embed_dim"] + lang_dim, ec["dim"])
        tech_dim = config.get("technique", {}).get("dim", 7)
        self.technique_proj = nn.Linear(tech_dim, pc["embed_dim"])
        nn.init.zeros_(self.technique_proj.weight); nn.init.zeros_(self.technique_proj.bias)

        self.encoder = ConformerEncoder(
            dim=ec["dim"], num_layers=ec["num_layers"], num_heads=ec["num_heads"],
            ff_mult=ec["ff_mult"], conv_kernel=ec["conv_kernel"],
            dropout=ec["dropout"], drop_path=ec.get("drop_path", 0.0))
        self.length_reg = LengthRegulator(
            dim=ec["dim"], pos_features=lrc["pos_features"], dur_features=lrc["dur_features"],
            pos_scale_init=lrc["pos_scale_init"], soft=lrc.get("soft", False),
            sigma_scale_init=lrc.get("sigma_scale_init", 0.2), sigma_min=lrc.get("sigma_min", 0.5))

        fmc = config["f0_fm"]
        self.fm = F0FlowDecoder(cond_dim=ec["dim"], dim=fmc.get("dim", 256),
                                n_blocks=fmc.get("n_blocks", 4), n_heads=fmc.get("n_heads", 4),
                                conv_kernel=fmc.get("conv_kernel", 7))
        self.cents_scale = fmc.get("cents_scale", 120.0)     # det/cents normalization (measured std ~121)
        self.cents_clip = fmc.get("cents_clip", 1200.0)      # deploy safety clamp (± octave)
        self.sample_steps = fmc.get("sample_steps", 20)
        self.voicing_gap_fill = fmc.get("voicing_gap_fill", 3)  # deploy: fill <=k-frame unvoiced gaps (brief 失声 fix)
        # deploy voicing: "note" = voiced where note>0 (照谱: pitch glides through consonants, no 失声; the model
        # voicing head over-unvoices consonant/SP edges + ACE's consonant-unvoicing is partly an RMVPE artifact)
        # vs "model" = the learned voicing head (+ gap-fill).
        self.voicing_mode = fmc.get("voicing_mode", "note")
        # S58-fix: det+FM-RESIDUAL. The pure FM-generate sampled the right MAGNITUDE but wrong VALUES (off-tune,
        # corr +0.33). So a DET head gives the in-tune conservative cents base (anchors pitch); the FM models only
        # the BOUNDED expression RESIDUAL (GT−det). Deploy cents = det + tau*residual; tau=0 = the stable in-tune det.
        self.residual_scale = fmc.get("residual_scale", 80.0)   # FM residual normalization (smaller than cents)
        self.tau = fmc.get("tau", 1.0)
        self.det_cents = nn.Sequential(                          # in-tune base: conservative cents mean
            nn.Conv1d(ec["dim"], 256, 5, padding=2), nn.SiLU(),
            nn.Conv1d(256, 256, 5, padding=2), nn.SiLU(), nn.Conv1d(256, 1, 1))

        # deterministic voicing head (content-driven; reads frame_hidden, NOT detached — its own encoder)
        self.voicing_head = nn.Sequential(
            nn.Conv1d(ec["dim"], 128, 5, padding=2), nn.SiLU(), nn.Conv1d(128, 1, 1))

        cdc = config.get("conditioning_dropout", {})
        self.spk_dropout = cdc.get("speaker", 0.0); self.lang_dropout = cdc.get("lang", 0.0)
        self.cond_dropout_on = True

    def encode(self, phonemes, note_pitch, phone_dur, note_dur, note_to_phone,
               speaker_id, lang_id=None, phone_mask=None, technique=None):
        B, N = phonemes.shape
        phone_emb = self.phone_embed(phonemes)
        if technique is not None:
            phone_emb = phone_emb + self.technique_proj(technique.float())
        pitch_emb = self.pitch_embed(note_pitch.clamp(0, 127))
        spk_emb = self.speaker_embed(speaker_id).unsqueeze(1).expand(-1, N, -1)
        if lang_id is None:
            lang_id = torch.zeros(B, dtype=torch.long, device=phonemes.device)
        lang_emb = self.language_embed(lang_id).unsqueeze(1).expand(-1, N, -1)
        if self.training and self.cond_dropout_on:
            if self.spk_dropout > 0:
                spk_emb = spk_emb * (torch.rand(B, device=phonemes.device) >= self.spk_dropout).float()[:, None, None]
            if self.lang_dropout > 0:
                lang_emb = lang_emb * (torch.rand(B, device=phonemes.device) >= self.lang_dropout).float()[:, None, None]
        x = self.input_proj(torch.cat([phone_emb, pitch_emb, spk_emb, lang_emb], dim=-1))
        x = self.encoder(x, phone_mask)
        frame_hidden, frame_mask, phone_idx, _ = self.length_reg(x, phone_dur, note_dur, note_to_phone)
        midi_frame = note_pitch.gather(1, phone_idx) * frame_mask
        return frame_hidden, frame_mask, midi_frame

    def det_cents_pred(self, frame_hidden):
        return self.det_cents(frame_hidden.transpose(1, 2)).squeeze(1)        # [B,T] raw cents (in-tune base)

    def det_loss(self, frame_hidden, cents_target, mask):
        """In-tune base: det cents L2 to GT cents (normalized) over clean voiced frames."""
        dc = self.det_cents_pred(frame_hidden)
        T = dc.shape[1]
        se = ((dc - cents_target[:, :T]) / self.cents_scale) ** 2
        m = mask[:, :T].float()
        return (se * m).sum() / (m.sum() + 1e-8)

    def fm_loss(self, frame_hidden, cents_target, mask):
        """x1-pred FM MSE on the normalized RESIDUAL (GT − det.detach()) over CLEAN voiced frames."""
        B, T, _ = frame_hidden.shape
        det = self.det_cents_pred(frame_hidden).detach()                     # [B,T] in-tune base, detached
        x1 = ((cents_target[:, :T] - det) / self.residual_scale).unsqueeze(1)  # [B,1,T] normalized residual
        t = torch.rand(B, device=frame_hidden.device)
        x0 = torch.randn_like(x1)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        x1hat = self.fm(xt, t, frame_hidden, mask)
        se = (x1hat - x1) ** 2
        m = mask[:, :T].unsqueeze(1).float()
        return (se * m).sum() / (m.sum() + 1e-8)

    def voicing_logit(self, frame_hidden):
        return self.voicing_head(frame_hidden.transpose(1, 2)).squeeze(1)     # [B,T]

    @torch.no_grad()
    def sample_cents(self, frame_hidden, mask=None, steps=None, tau=None):
        """cents = in-tune det base + tau * FM expression residual. tau=0 = the stable in-tune det alone."""
        steps = steps or self.sample_steps
        tau = self.tau if tau is None else tau
        B, T, _ = frame_hidden.shape
        det = self.det_cents_pred(frame_hidden)                              # [B,T] in-tune base
        x = torch.randn(B, 1, T, device=frame_hidden.device)
        for k in range(steps):
            tv = k / steps
            t = torch.full((B,), tv, device=x.device)
            x1hat = self.fm(x, t, frame_hidden, mask)
            v = (x1hat - x) / max(1.0 - tv, 1.0 / steps)
            x = x + v * (1.0 / steps)
        resid = x[:, 0] * self.residual_scale                               # [B,T] bounded expression residual
        return (det + tau * resid).clamp(-self.cents_clip, self.cents_clip)  # [B,T] cents

    @torch.no_grad()
    def infer_f0(self, frame_hidden, frame_mask, midi_frame, steps=None, tau=None):
        cents = self.sample_cents(frame_hidden, frame_mask, steps, tau)
        if self.voicing_mode == "note":          # 照谱: voiced through notes, silent at rests (continuous pitch)
            voiced = midi_frame > 0
        else:                                     # learned head + brief-gap-fill
            voiced = torch.sigmoid(self.voicing_logit(frame_hidden)) > 0.5
            if self.voicing_gap_fill > 0:
                vn = (voiced & frame_mask).cpu().numpy()
                for b in range(vn.shape[0]):
                    vn[b] = _fill_voicing_gaps(vn[b], self.voicing_gap_fill)
                voiced = torch.from_numpy(vn).to(frame_hidden.device)
        voiced = voiced & frame_mask
        midi_hz = 440.0 * 2.0 ** ((midi_frame.float() - 69.0) / 12.0) * (midi_frame > 0)
        f0 = midi_hz * (2.0 ** (cents / 1200.0)) * voiced.float() * frame_mask.float()
        return f0, voiced

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
