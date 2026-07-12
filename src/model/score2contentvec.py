"""Score2ContentVec v3 — musical score to CONTINUOUS ContentVec (generative) + F0.

Pipeline: phonemes + pitch + duration + speaker → frame_hidden → flow-matching decoder
          samples continuous ContentVec [T,768]; F0/voicing predicted deterministically.
Downstream: So-VITS-SVC 4.1 (continuous ContentVec is SoVITS-native — no codebook).
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .modules import (ConformerEncoder, ConformerBlock, RelativePositionalEncoding,
                      MultiHeadAttention)


class LengthRegulator(nn.Module):
    """Expand phone-level representations to frame-level using durations.
    Adds intra-phone/note position and log-duration features.
    Fully vectorized — no Python loops over batch or phone dimensions."""

    def __init__(self, dim, pos_features=True, dur_features=True, pos_scale_init=3.0,
                 soft=False, sigma_scale_init=0.2, sigma_min=0.5):
        super().__init__()
        self.pos_features = pos_features
        self.dur_features = dur_features
        self.soft = soft
        if soft:
            self.log_sigma_scale = nn.Parameter(torch.tensor(math.log(sigma_scale_init)))
            self.sigma_min = sigma_min

        extra_dim = 0
        if pos_features:
            extra_dim += 2
        if dur_features:
            extra_dim += 2

        if extra_dim > 0:
            self.feat_proj = nn.Linear(extra_dim, dim)
            nn.init.zeros_(self.feat_proj.weight)
            nn.init.zeros_(self.feat_proj.bias)
            self.pos_scale = nn.Parameter(torch.tensor(pos_scale_init))
        else:
            self.feat_proj = None

    def forward(self, phone_hidden, phone_dur, note_dur, note_to_phone):
        B, N, D = phone_hidden.shape
        T = phone_dur.sum(dim=1).max().item()

        cum_dur = phone_dur.cumsum(dim=1)
        frame_pos = torch.arange(T, device=phone_dur.device)

        phone_idx = torch.searchsorted(
            cum_dur.contiguous(), frame_pos.unsqueeze(0).expand(B, -1).contiguous(),
            right=True,
        )
        phone_idx = phone_idx.clamp(max=N - 1)

        total_dur = phone_dur.sum(dim=1)
        frame_mask = frame_pos.unsqueeze(0) < total_dur.unsqueeze(1)

        if self.soft:
            # Soft Gaussian upsampling: each frame is a blend of nearby phones,
            # robust to +-frame boundary jitter, models coarticulation transitions.
            # phone_idx (hard) is still returned/used for position features + F0 midi.
            phone_start_c = torch.zeros_like(cum_dur)
            phone_start_c[:, 1:] = cum_dur[:, :-1]
            centers = phone_start_c.float() + phone_dur.float() / 2           # [B,N]
            sigma = (torch.exp(self.log_sigma_scale) * phone_dur.float()).clamp(min=self.sigma_min)
            dist = frame_pos.view(1, -1, 1).float() - centers.unsqueeze(1)    # [B,T,N]
            logits = -(dist * dist) / (2.0 * sigma.unsqueeze(1) ** 2)
            logits = logits.masked_fill((phone_dur == 0).unsqueeze(1), float("-inf"))
            weights = torch.softmax(logits, dim=-1)                           # [B,T,N]
            frame_hidden = torch.bmm(weights.to(phone_hidden.dtype), phone_hidden)
        else:
            frame_hidden = phone_hidden.gather(
                1, phone_idx.unsqueeze(-1).expand(-1, -1, D),
            )

        pos_targets = None
        if self.feat_proj is not None:
            phone_start = torch.zeros_like(cum_dur)
            phone_start[:, 1:] = cum_dur[:, :-1]

            feats = []
            if self.pos_features:
                start_f = phone_start.float().gather(1, phone_idx)
                dur_f = phone_dur.float().gather(1, phone_idx).clamp(min=1)
                pos_phone = (frame_pos.unsqueeze(0).float() - start_f) / dur_f

                pos_note = self._pos_in_note(
                    phone_dur, note_dur, note_to_phone,
                    phone_idx, phone_start, frame_pos,
                )
                feats.extend([pos_phone, pos_note])

            if self.dur_features:
                feats.append(torch.log1p(phone_dur.float()).gather(1, phone_idx))
                feats.append(torch.log1p(note_dur.float()).gather(1, phone_idx))

            feat_tensor = torch.stack(feats, dim=-1)
            proj = self.feat_proj(feat_tensor) * self.pos_scale
            frame_hidden = frame_hidden + proj * frame_mask.unsqueeze(-1).float()
            if self.pos_features:
                pos_targets = torch.stack([pos_phone, pos_note], dim=-1)   # [B,T,2]

        return frame_hidden, frame_mask, phone_idx, pos_targets

    def _pos_in_note(self, phone_dur, note_dur, note_to_phone,
                     phone_idx, phone_start, frame_pos):
        B, N = phone_dur.shape

        note_boundary = torch.ones(B, N, dtype=torch.bool, device=phone_dur.device)
        note_boundary[:, 1:] = note_to_phone[:, 1:] != note_to_phone[:, :-1]

        global_cum = phone_start.float()
        group_id = note_boundary.long().cumsum(dim=1) - 1

        boundary_cum = torch.zeros(B, N, device=phone_dur.device)
        boundary_cum.scatter_(1, group_id, global_cum * note_boundary.float(),
                              reduce='add')

        start_in_note = global_cum - boundary_cum.gather(1, group_id)

        start_in_note_f = start_in_note.gather(1, phone_idx)
        phone_start_f = phone_start.float().gather(1, phone_idx)
        pos_within_phone = frame_pos.unsqueeze(0).float() - phone_start_f
        pos_total = start_in_note_f + pos_within_phone

        note_dur_f = note_dur.float().gather(1, phone_idx).clamp(min=1)
        return pos_total / note_dur_f


class F0Predictor(nn.Module):
    """Predicts F0 as cents deviation from MIDI note pitch."""

    def __init__(self, input_dim, hidden_dim=256, lstm_layers=2, conv_channels=128,
                 conv_kernel=5, dropout=0.2, cents_bound=1200.0, tech_dim=7):
        super().__init__()
        self.cents_bound = cents_bound       # Run6: ±octave margin beyond the 3-note pitch window
        # Run6: technique conditioning ON THE F0 HEAD (vibrato/glissando/falsetto reshape the F0
        # contour). Zero-init -> no-op at start; trained by F0 loss; lets the F0 head de-average
        # e.g. vibrato-F0 from smooth-F0. Separate from the token-path technique_proj.
        self.tech_proj = nn.Linear(tech_dim, input_dim)
        nn.init.zeros_(self.tech_proj.weight)
        nn.init.zeros_(self.tech_proj.bias)
        self.pre_proj = nn.Linear(input_dim + 1, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, conv_channels, conv_kernel, padding=conv_kernel // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, conv_kernel, padding=conv_kernel // 2),
            nn.SiLU(),
        )
        self.cents_out = nn.Linear(conv_channels, 1)
        self.voicing_out = nn.Linear(conv_channels, 1)

    def forward(self, frame_hidden, midi_pitch, frame_mask, note_lo=None, note_hi=None, technique_frame=None):
        """
        note_lo/note_hi: [B,T] min/max MIDI of the {prev,cur,next}-note window (Run6). When given,
        the cents output is bounded to that window ± cents_bound (1 octave) relative to the current
        note, so legato/glissando frames (which lie INSIDE the window) are never clipped.
        Returns: f0_hz [B,T], voicing_logit [B,T], f0_cents [B,T] (cents dev from current note).
        """
        pitched = (midi_pitch > 0).float()
        midi_hz = 440.0 * (2 ** ((midi_pitch.float() - 69) / 12)) * pitched
        midi_feat = midi_hz.unsqueeze(-1) / 1000.0

        if technique_frame is not None:          # condition F0 on technique (vibrato/glissando/falsetto…)
            frame_hidden = frame_hidden + self.tech_proj(technique_frame)
        x = self.drop(self.pre_proj(torch.cat([frame_hidden, midi_feat], dim=-1)))
        x, _ = self.lstm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)

        raw = self.cents_out(x).squeeze(-1)
        if note_lo is not None:
            # asymmetric bound to the 3-note window ± cents_bound; piecewise-tanh centered at 0
            # (= current-note pitch when raw=0), saturating to lo_c (raw<0) / hi_c (raw>0).
            hi_c = ((note_hi - midi_pitch).float() * 100.0 + self.cents_bound).clamp(min=1.0)
            lo_c = ((note_lo - midi_pitch).float() * 100.0 - self.cents_bound).clamp(max=-1.0)
            f0_cents = torch.where(raw >= 0, hi_c * torch.tanh(raw / hi_c),
                                   (-lo_c) * torch.tanh(raw / (-lo_c))) * pitched
        else:
            f0_cents = torch.tanh(raw) * self.cents_bound * pitched
        f0_hz = midi_hz * (2 ** (f0_cents / 1200))

        voicing_logit = self.voicing_out(x).squeeze(-1)
        return f0_hz, voicing_logit, f0_cents


class DiTBlock(nn.Module):
    """DiT-style velocity block. Frame-level FiLM (scale/shift) from the frame-aligned condition
    (the score-derived conditioning c), adaLN-zero gates from the FM timestep (gates zero-init → block starts
    ≈ identity = stable from-scratch training), self-attention (cross-frame coherence) + depthwise
    conv + MLP (local detail). Replaces Run8's weak additive conv(h+c) conditioning."""

    def __init__(self, dim, n_heads, conv_kernel=15, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.dw = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.film = nn.Linear(dim, 4 * dim)          # frame-level (scale,shift)×2 sublayers, from cond
        self.gate = nn.Linear(dim, 2 * dim)          # adaLN-zero gates from t (global per-sample)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h, cond, t_emb, mask):
        # h, cond: [B,T,dim]; t_emb: [B,dim]; mask: [B,T] bool
        sc1, sh1, sc2, sh2 = self.film(cond).chunk(4, dim=-1)              # [B,T,dim] each
        g1, g2 = torch.tanh(self.gate(t_emb))[:, None, :].chunk(2, dim=-1)  # [B,1,dim]; S26: tanh-BOUND the
        # adaLN gate too. The gate is zero-init (block=identity at start, tanh(0)=0 keeps that) but UNBOUNDED —
        # it grows to turn the block on and CAN blow up the multiplicative g*attn / g*ff residual = the slow-
        # accumulation that tipped r10b @36K even WITH scale-tanh+clip (scale-tanh delayed 27K->36K, not enough).
        # S26: tanh-BOUND the FiLM scale -> (1+tanh) in [0,2] (ample). Unbounded (1+sc) let the scale grow
        # during training -> activation/gradient blow-up -> r10 DIVERGED @~27K (gnorm spike 18; voicing +
        # sample crashed via a corrupted encoder). SAME root cause as r9b's fp16 NaN @32K (bf16 dodged the
        # NaN but not the instability; c-only conditioning makes FiLM blow up more readily than r9b's c+z).
        x = self.norm1(h) * (1 + torch.tanh(sc1)) + sh1
        h = h + g1 * self.attn(x, mask)
        x = self.norm2(h) * (1 + torch.tanh(sc2)) + sh2
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)                     # local mixing
        h = h + g2 * self.ff(F.silu(x))
        return h


class FlowMatchingDecoder(nn.Module):
    """x1-prediction flow-matching velocity net over CONTINUOUS contentvec [T,768], conditioned on the
    SCORE (frame_hidden c). Clean cv|c FM (S26): no realization latent — the model must learn the whole
    conditional p(cv|c) and SAMPLE a sharp realization from it. 3 de-risked ingredients: x1-prediction,
    ABSOLUTE position, per-dim-normalized target (parent's cv_mean/cv_std). CFG: a learned null for c."""

    def __init__(self, cond_dim, dim=512, n_blocks=8, n_heads=8, dropout=0.0,
                 conv_kernel=15, grad_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.grad_checkpoint = grad_checkpoint     # recompute block activations in backward (saves VRAM)
        self.in_proj = nn.Conv1d(768, dim, 1)                              # x_t -> h
        self.cond_in = nn.Linear(cond_dim, dim)                          # score c -> frame conditioning
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))            # CFG null (learned)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads, conv_kernel, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_proj = nn.Conv1d(dim, 768, 1)

    def _time_emb(self, t):                                                # [B] -> [B,dim]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None]
        return self.time_mlp(torch.cat([a.sin(), a.cos()], -1))

    def _pos_emb(self, T, dev):                                            # -> [1,T,dim]
        half = self.dim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=dev) / half)
        a = torch.arange(T, device=dev)[:, None] * fr[None]
        return torch.cat([a.sin(), a.cos()], -1).unsqueeze(0)

    def forward(self, x_t, t, c, mask=None, drop=None):
        """x_t [B,768,T]; t [B] in [0,1]; c [B,T,cond_dim] (score); drop [B] bool -> replace c with the
        learned null (CFG). Returns x1_hat [B,768,T]."""
        B, _, T = x_t.shape
        cc = c
        if drop is not None:
            null = self.null_cond.view(1, 1, -1).expand(B, T, -1)
            cc = torch.where(drop[:, None, None], null, c)
        cond = self.cond_in(cc) + self._pos_emb(T, x_t.device)                                # [B,T,dim]
        t_emb = self._time_emb(t)                                          # [B,dim]
        h = self.in_proj(x_t).transpose(1, 2)                            # [B,T,dim]
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                h = checkpoint(blk, h, cond, t_emb, mask, use_reentrant=False)
            else:
                h = blk(h, cond, t_emb, mask)
        return self.out_proj(self.out_norm(h).transpose(1, 2))           # [B,768,T] = x1_hat (the RESIDUAL, S27)


class DetHead(nn.Module):
    """Deterministic conditional MEAN E[cv|c] in NORMALIZED cv space (S27 det_head+FM-residual fix).
    The 'close' base / data-achievable FLOOR: Probe D proved a head exactly like this on the frozen
    encoder reaches a UNIFORM per-lang cos 0.68-0.79 (incl ko 0.68) — det_head is robust to the
    speaker/lang isolation that defeats the FM. The FM then SAMPLES the sharp realization RESIDUAL on
    top (output = μ + τ·residual): close (det_head) AND sharp (FM), with the floor as the τ=0 safety net.
    Conv stack = a fair, proven mean-regressor (mirrors scripts/probe_d_dethead.py)."""

    def __init__(self, cond_dim, out=768, hidden=512, k=5, n_layers=3):
        super().__init__()
        layers = [nn.Conv1d(cond_dim, hidden, k, padding=k // 2), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv1d(hidden, hidden, k, padding=k // 2), nn.SiLU()]
        layers += [nn.Conv1d(hidden, out, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, c):                                  # c [B,T,cond_dim] -> [B,T,768] normalized cv mean
        return self.net(c.transpose(1, 2)).transpose(1, 2)


class Score2ContentVec(nn.Module):
    """Full model: musical score + speaker → continuous ContentVec (det_head mean + FM residual) + F0."""

    SP_ID = 3

    def __init__(self, config):
        super().__init__()
        pc = config["phoneme"]
        sc = config["speaker"]
        ec = config["encoder"]
        lrc = config["length_regulator"]
        fc = config["f0_predictor"]
        fmc = config["fm_decoder"]

        self.f0_detach = fc.get("detach_encoder", False)

        lc = config.get("language", {})
        lang_dim = lc.get("embed_dim", 64)
        self.phone_embed = nn.Embedding(pc["vocab_size"], pc["embed_dim"], padding_idx=pc["padding_idx"])
        self.pitch_embed = nn.Embedding(128, pc["embed_dim"])
        self.speaker_embed = nn.Embedding(sc["n_speakers"], sc["embed_dim"])
        self.language_embed = nn.Embedding(lc.get("n_langs", 9), lang_dim)   # Run9 (β): language conditioning
        self.input_proj = nn.Linear(pc["embed_dim"] * 2 + sc["embed_dim"] + lang_dim, ec["dim"])

        # Per-phone technique conditioning (Run 6, S14): zero-init -> starts as a NO-OP (identical
        # to the Run-5 architecture), learns technique deltas; ADDED to phone_emb so a "breathy /a/"
        # shifts off the "normal /a/" embedding = direct de-averaging. technique: [B,N,tech_dim] multi-hot.
        tech_dim = config.get("technique", {}).get("dim", 7)
        self.technique_proj = nn.Linear(tech_dim, pc["embed_dim"])
        nn.init.zeros_(self.technique_proj.weight)
        nn.init.zeros_(self.technique_proj.bias)

        self.encoder = ConformerEncoder(
            dim=ec["dim"],
            num_layers=ec["num_layers"],
            num_heads=ec["num_heads"],
            ff_mult=ec["ff_mult"],
            conv_kernel=ec["conv_kernel"],
            dropout=ec["dropout"],
            drop_path=ec.get("drop_path", 0.0),
        )

        self.length_reg = LengthRegulator(
            dim=ec["dim"],
            pos_features=lrc["pos_features"],
            dur_features=lrc["dur_features"],
            pos_scale_init=lrc["pos_scale_init"],
            soft=lrc.get("soft", False),
            sigma_scale_init=lrc.get("sigma_scale_init", 0.2),
            sigma_min=lrc.get("sigma_min", 0.5),
        )

        self.f0_pred = F0Predictor(
            input_dim=ec["dim"],
            hidden_dim=fc["hidden_dim"],
            lstm_layers=fc["lstm_layers"],
            conv_channels=fc["conv_channels"],
            conv_kernel=fc["conv_kernel"],
            dropout=fc["dropout"],
            cents_bound=fc.get("cents_bound", 1200.0),
            tech_dim=config.get("technique", {}).get("dim", 7),
        )

        # S26: CLEAN cv|c FM — NO realization latent (cVAE posterior/prior/FM-prior all removed; that line
        # was a detour, S24). The decoder must learn the whole conditional p(cv|c) and SAMPLE a sharp
        # realization from it = the honest baseline + the vehicle to diagnose why our FM under-disperses
        # toward the central 啊 (the open "为什么" — see project-token-rootcause / session26).
        # Generative x1-prediction FM decoder over CONTINUOUS contentvec, conditioned on the score c only.
        self.fm_decoder = FlowMatchingDecoder(
            cond_dim=ec["dim"],
            dim=fmc.get("dim", 512),
            n_blocks=fmc.get("n_blocks", 8),
            n_heads=fmc.get("n_heads", 8),
            dropout=fmc.get("dropout", 0.0),
            grad_checkpoint=fmc.get("grad_checkpoint", False),
        )
        # S51: optional F0 CONDITIONING — fuse the actual F0 contour (log-f0 + cents-from-note + voiced) INTO
        # frame_hidden (the shared det_head/FM conditioning). The model is otherwise F0-BLIND (F0 is a detached
        # head); the controlled ambiguity diag shows F0 resolves ~16-20% of the hard-frame one-to-many, and with
        # a DETERMINISTIC teacher (DiffSinger-cv) it closes the collision -> det_head mean becomes sharp. DiffSinger-
        # style fused (F0+content) conditioning. Flag-gated (default off = backward compatible w/ old checkpoints).
        self.f0_condition = fmc.get("f0_condition", False)
        if self.f0_condition:
            self.f0_cond = nn.Sequential(nn.Linear(3, ec["dim"]), nn.SiLU(), nn.Linear(ec["dim"], ec["dim"]))
        # S52 lever A: make the (otherwise MIDI-only, F0-BLIND) ENCODER F0-aware — add per-phone F0 stats
        # [mean cents-from-note, std cents (vibrato/scoop magnitude), voiced frac] to the encoder input so the
        # phone-context KNOWS about scoops/vibrato (the concrete input-gap vs DiffSinger, whose acoustic model is
        # F0-conditioned throughout). zero-init -> no-op at start = backward-compatible. EMPIRICAL try (the
        # held-vowel 塌 mystery is unsolved; no metric can guide this — EAR only).
        self.encoder_f0 = fmc.get("encoder_f0", False)
        if self.encoder_f0:
            self.enc_f0_proj = nn.Linear(3, ec["dim"])
            nn.init.zeros_(self.enc_f0_proj.weight); nn.init.zeros_(self.enc_f0_proj.bias)
        self.cfg_dropout = fmc.get("cfg_dropout", 0.1)        # CFG: train-time null-cond probability
        # per-dim contentvec normalization (computed over TRAIN only, compute_cv_norm.py)
        norm_path = Path(fmc.get("norm_path", "processed/contentvec_norm.npz"))
        if norm_path.exists():
            nz = np.load(norm_path)
            self.register_buffer("cv_mean", torch.tensor(nz["mean"], dtype=torch.float32))
            self.register_buffer("cv_std", torch.tensor(nz["std"], dtype=torch.float32))
        else:                                                  # fallback (smoke before stats exist)
            print(f"[WARN] {norm_path} missing -> contentvec NOT normalized (run compute_cv_norm.py "
                  "before training; normalization is a required FM ingredient)")
            self.register_buffer("cv_mean", torch.zeros(768))
            self.register_buffer("cv_std", torch.ones(768))

        # S27 det_head+FM-residual fix. det_head = the conditional MEAN (the floor); the FM (above) now
        # models the RESIDUAL cv−μ (a smaller, cross-lingually SHARED realization texture → less data-hungry,
        # the root fix for the FM scatter on data-poor langs). output = μ + τ·residual.
        self.fm_residual = fmc.get("residual", False)
        self.tau = fmc.get("tau", 1.0)                        # inference residual temperature (0=det_head floor)
        self.det_head = DetHead(cond_dim=ec["dim"])
        # per-dim RESIDUAL normalization (the residual is smaller-scale than cv → its OWN per-dim-norm = the
        # 3rd FM ingredient applied to the residual; computed AFTER stage-1 by compute_residual_norm.py)
        rnorm_path = Path(fmc.get("residual_norm_path", "processed/residual_norm.npz"))
        if self.fm_residual and rnorm_path.exists():
            rz = np.load(rnorm_path)
            self.register_buffer("res_std", torch.tensor(rz["std"], dtype=torch.float32))
        else:                                                  # fallback (stage-1 / smoke: unit scale)
            self.register_buffer("res_std", torch.ones(768))

        # S27 de-isolate: independent train-time DROPOUT of the speaker-embed (the DOMINANT 256-d isolator;
        # singer→lang INJECTIVE, zh=50 speakers vs ru=1) + lang-embed → the model learns a phone+context-driven
        # frame_hidden that POOLS all data (data-poor langs borrow the shared phone→cv). Probe E/G: speaker/lang
        # are minor perturbations of the SHARED phone region (xlang 0.86 ≥ xsinger 0.83), so pooling fixes the
        # hunger; ɛ/s/f are lang-specific → DROPOUT (not drop) keeps lang for the data-rich case. Both det_head
        # and FM read this same de-isolated frame_hidden (one conditioning — no left/right-brain split).
        cdc = config.get("conditioning_dropout", {})
        self.spk_dropout = cdc.get("speaker", 0.0)
        self.lang_dropout = cdc.get("lang", 0.0)
        self.stage = config.get("training", {}).get("stage", 1)
        self.cond_dropout_on = True            # set_stage() toggles: ON in stage-1 (de-isolate the encoder),
                                               # OFF in stage-2 (frozen encoder → deterministic gated frame_hidden)

        # Frame-level decoder: self-attention over frames to contextualize each
        # frame (denoise alignment jitter, fix boundary-bleed, phrase-level F0).
        dec = config.get("decoder", {})
        self.dec_layers = dec.get("num_layers", 0)
        self.dec_checkpoint = dec.get("grad_checkpoint", False)
        if self.dec_layers > 0:
            self.dec_pos_enc = RelativePositionalEncoding(ec["dim"])
            self.decoder = nn.ModuleList([
                ConformerBlock(
                    ec["dim"], dec.get("num_heads", ec["num_heads"]),
                    dec.get("ff_mult", 4), dec.get("conv_kernel", 15),
                    dec.get("dropout", ec["dropout"]), dec.get("drop_path", 0.0),
                    use_conv=dec.get("use_conv", False),   # pure attention by default (conv is the slow part)
                )
                for _ in range(self.dec_layers)
            ])
        else:
            self.decoder = None

        # pos_aux: force the model to keep position features decodable
        # (v2 found pos features go dead without this auxiliary loss).
        use_pos_aux = config["training"].get("pos_aux_weight", 0.0) > 0 and lrc["pos_features"]
        self.pos_head = nn.Linear(ec["dim"], 2) if use_pos_aux else None

    def _note_window(self, note_pitch, note_to_phone):
        """Per-phone [lo, hi] = min/max MIDI over non-rest of the {prev, cur, next} NOTE (Run6).
        Legato/glissando frames (transitioning between adjacent notes) lie inside [lo,hi]; used to
        bound F0 output + mask dirty F0 targets WITHOUT clipping/dropping those transitions.
        Vectorized: scatter per-phone pitch into note slots (amax so rest/pad 0s don't clobber)."""
        B, N = note_pitch.shape
        note_p = torch.zeros(B, N + 2, device=note_pitch.device, dtype=note_pitch.dtype)
        idx = (note_to_phone + 1).clamp(0, N + 1)
        note_p.scatter_reduce_(1, idx, note_pitch, reduce="amax", include_self=True)
        g = note_to_phone
        cur = note_p.gather(1, g + 1)
        prev = note_p.gather(1, g)                          # slot g   = note g-1 (0 at the edge)
        nxt = note_p.gather(1, (g + 2).clamp(max=N + 1))    # slot g+2 = note g+1
        stack = torch.stack([cur, prev, nxt], dim=0).float()
        pos = stack > 0                                     # rests (0) excluded from the window
        lo = stack.masked_fill(~pos, float("inf")).amin(dim=0)
        hi = stack.masked_fill(~pos, float("-inf")).amax(dim=0)
        cur_f = cur.float()
        lo = torch.where(cur > 0, lo, cur_f)                # rest frames: lo=hi=cur (excluded anyway)
        hi = torch.where(cur > 0, hi, cur_f)
        return lo, hi

    def _phone_f0_stats(self, f0, phone_dur, note_pitch):
        """S52 lever A: per-phone F0 stats for the F0-aware encoder. Returns [B,N,3] =
        [mean cents-from-note /200, std cents (vibrato/scoop magnitude) /200, voiced frac]. Computed from the
        given f0 (GT at train / the user's pitch curve at infer) by segmenting frame-level f0 into phones."""
        B, N = phone_dur.shape
        T = int(phone_dur.sum(dim=1).max().item())
        f0 = f0[:, :T]
        cum = phone_dur.cumsum(dim=1)
        frame_pos = torch.arange(T, device=f0.device)
        fphone = torch.searchsorted(cum.contiguous(),
                                    frame_pos.unsqueeze(0).expand(B, -1).contiguous(), right=True).clamp(max=N - 1)
        note_hz = 440.0 * 2.0 ** ((note_pitch.float() - 69.0) / 12.0) * (note_pitch > 0)        # [B,N]
        note_hz_fr = note_hz.gather(1, fphone)                                                  # [B,T]
        vd = (f0 > 30).float()
        cents = torch.where((f0 > 30) & (note_hz_fr > 0),
                            1200.0 * torch.log2(f0.clamp(min=1e-6) / note_hz_fr.clamp(min=1e-6)),
                            torch.zeros_like(f0)).clamp(-1200.0, 1200.0)
        z = lambda: torch.zeros(B, N, device=f0.device, dtype=f0.dtype)
        cnt = z().scatter_add(1, fphone, torch.ones_like(f0))
        vcnt = z().scatter_add(1, fphone, vd)
        csum = z().scatter_add(1, fphone, cents * vd)
        csq = z().scatter_add(1, fphone, (cents * cents) * vd)
        cmean = csum / vcnt.clamp(min=1)
        cstd = torch.sqrt((csq / vcnt.clamp(min=1) - cmean * cmean).clamp(min=0))
        vfrac = vcnt / cnt.clamp(min=1)
        return torch.stack([cmean / 200.0, cstd / 200.0, vfrac], dim=-1)                        # [B,N,3]

    def forward(self, phonemes, note_pitch, phone_dur, note_dur, note_to_phone,
                speaker_id, lang_id=None, phone_mask=None, technique=None, f0=None):
        """
        Args:
            phonemes: [B, N] phoneme IDs
            note_pitch: [B, N] MIDI note numbers
            phone_dur: [B, N] frames per phone
            note_dur: [B, N] frames per note
            note_to_phone: [B, N] note index each phone belongs to
            speaker_id: [B] speaker index
            phone_mask: [B, N] valid phone mask
        Returns:
            dict with 'frame_hidden' [B,T,D] (FM conditioning), 'f0'/'voicing_logit' [B,T], 'frame_mask' [B,T]
        """
        B, N = phonemes.shape
        phone_emb = self.phone_embed(phonemes)
        if technique is not None:
            phone_emb = phone_emb + self.technique_proj(technique.float())
        pitch_emb = self.pitch_embed(note_pitch.clamp(0, 127))
        spk_emb = self.speaker_embed(speaker_id).unsqueeze(1).expand(-1, N, -1)
        if lang_id is None:
            lang_id = torch.zeros(B, dtype=torch.long, device=phonemes.device)
        lang_emb = self.language_embed(lang_id).unsqueeze(1).expand(-1, N, -1)
        if self.training and self.cond_dropout_on:   # S27 de-isolate: per-sample dropout of the spk / lang embeds
            if self.spk_dropout > 0:
                keep = (torch.rand(B, device=phonemes.device) >= self.spk_dropout).float()[:, None, None]
                spk_emb = spk_emb * keep
            if self.lang_dropout > 0:
                keep = (torch.rand(B, device=phonemes.device) >= self.lang_dropout).float()[:, None, None]
                lang_emb = lang_emb * keep
        x = self.input_proj(torch.cat([phone_emb, pitch_emb, spk_emb, lang_emb], dim=-1))

        if self.encoder_f0 and f0 is not None:        # S52 lever A: F0-aware encoder (per-phone scoop/vibrato stats)
            ef0 = self.enc_f0_proj(self._phone_f0_stats(f0, phone_dur, note_pitch))
            if phone_mask is not None:
                ef0 = ef0 * phone_mask.unsqueeze(-1).float()
            x = x + ef0

        x = self.encoder(x, phone_mask)

        frame_hidden, frame_mask, phone_idx, pos_targets = self.length_reg(
            x, phone_dur, note_dur, note_to_phone)

        # pos_aux reads the position-injection point (before the decoder)
        pos_pred = torch.sigmoid(self.pos_head(frame_hidden)) if self.pos_head is not None else None

        # frame-level decoder: contextualize frames (denoise jitter, boundary-bleed, phrase F0)
        if self.decoder is not None:
            h = self.dec_pos_enc(frame_hidden)
            for layer in self.decoder:
                if self.dec_checkpoint and self.training:
                    h = checkpoint(layer, h, frame_mask, use_reentrant=False)
                else:
                    h = layer(h, frame_mask)
            frame_hidden = h

        midi_frame = note_pitch.gather(1, phone_idx) * frame_mask
        # 3-note (prev/cur/next) pitch window for the F0 octave bound + loss mask (Run6: preserves
        # legato/glissando — transition frames lie inside [lo,hi], so aren't clipped or dropped).
        nlo, nhi = self._note_window(note_pitch, note_to_phone)
        frame_note_lo = nlo.gather(1, phone_idx) * frame_mask
        frame_note_hi = nhi.gather(1, phone_idx) * frame_mask

        technique_frame = None
        if technique is not None:
            tdim = technique.shape[-1]
            technique_frame = technique.gather(
                1, phone_idx.unsqueeze(-1).expand(-1, -1, tdim)) * frame_mask.unsqueeze(-1)
        f0_input = frame_hidden.detach() if self.f0_detach else frame_hidden
        f0_hz, voicing_logit, f0_cents = self.f0_pred(
            f0_input, midi_frame, frame_mask, frame_note_lo, frame_note_hi, technique_frame)

        if self.f0_condition:                          # S51: fuse F0 into the FM/det_head conditioning
            Tf = frame_hidden.shape[1]
            f0c = f0[:, :Tf] if f0 is not None else f0_hz   # GT f0 (train, sliced to T) or predicted f0 (infer)
            vd = (f0c > 30).float()
            lf0 = (torch.log(f0c.clamp(min=1.0)) - 5.4) / 0.5 * vd
            note_hz = 440.0 * 2.0 ** ((midi_frame.float() - 69.0) / 12.0)
            cents = torch.where((f0c > 30) & (midi_frame > 0),
                                1200.0 * torch.log2(f0c.clamp(min=1e-6) / note_hz.clamp(min=1e-6)),
                                torch.zeros_like(f0c)).clamp(-1200.0, 1200.0)
            f0feat = torch.stack([lf0, cents / 200.0, vd], dim=-1)               # [B,T,3]
            frame_hidden = frame_hidden + self.f0_cond(f0feat) * frame_mask.unsqueeze(-1)

        return {
            "frame_hidden": frame_hidden,        # conditioning for the FM decoder
            "f0": f0_hz,
            "f0_cents": f0_cents,
            "voicing_logit": voicing_logit,
            "frame_mask": frame_mask,
            "frame_note_lo": frame_note_lo,
            "frame_note_hi": frame_note_hi,
            "pos_pred": pos_pred,
            "pos_targets": pos_targets,
        }

    def det_loss(self, frame_hidden, target_cv, mask):
        """STAGE-1 objective: det_head L2 to the normalized cv = the conditional MEAN E[cv|c] (the floor).
        Trains encoder + det_head (+ F0 separately). target_cv [B,T,768] raw; mask [B,T] bool."""
        B, T, _ = frame_hidden.shape
        x1n = (target_cv[:, :T] - self.cv_mean) / self.cv_std                 # [B,T,768] normalized cv
        mu = self.det_head(frame_hidden)
        se = (mu - x1n) ** 2
        m = mask.unsqueeze(-1).float()
        return (se * m).sum() / (m.sum() * 768 + 1e-8)

    def fm_loss(self, frame_hidden, target_cv, mask):
        """Flow-matching x1-prediction MSE over valid frames (incl. SP/AP — silence/breath contentvec is
        real and worth learning), conditioned on the score c. target_cv [B,T,768] raw; mask [B,T] bool.
        S27: when fm_residual, the FM target is the NORMALIZED RESIDUAL (x1−μ)/res_std (μ=det_head, DETACHED —
        the FM never trains det_head; in stage-2 det_head is also frozen). Else (legacy) it models cv directly."""
        B, T, _ = frame_hidden.shape
        target_cv = target_cv[:, :T]
        x1n = (target_cv - self.cv_mean) / self.cv_std                        # [B,T,768] normalized cv
        if self.fm_residual:
            mu = self.det_head(frame_hidden).detach()                        # normalized conditional mean
            tgt = ((x1n - mu) / self.res_std).transpose(1, 2)                # [B,768,T] normalized RESIDUAL
        else:
            tgt = x1n.transpose(1, 2)                                         # [B,768,T] normalized cv
        t = torch.rand(B, device=tgt.device)
        x0 = torch.randn_like(tgt)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * tgt
        drop = torch.rand(B, device=tgt.device) < self.cfg_dropout           # CFG null-cond dropout
        x1hat = self.fm_decoder(xt, t, frame_hidden, mask, drop)
        se = ((x1hat - tgt) ** 2).transpose(1, 2)                            # [B,T,768]
        m = mask.unsqueeze(-1).float()
        return (se * m).sum() / (m.sum() * 768 + 1e-8)

    @torch.no_grad()
    def fm_per_sample(self, frame_hidden, target_cv, mask):
        """Per-sample FM MSE [B] (no CFG drop), conditioned on the score c — for per-language val
        monitoring (the aggregate masks minority-lang overfitting; feedback-per-lang-val)."""
        B, T, _ = frame_hidden.shape
        target_cv = target_cv[:, :T]
        x1n = (target_cv - self.cv_mean) / self.cv_std
        if self.fm_residual:
            mu = self.det_head(frame_hidden)
            tgt = ((x1n - mu) / self.res_std).transpose(1, 2)
        else:
            tgt = x1n.transpose(1, 2)
        t = torch.rand(B, device=tgt.device)
        x0 = torch.randn_like(tgt)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * tgt
        x1hat = self.fm_decoder(xt, t, frame_hidden, mask, None)
        se = ((x1hat - tgt) ** 2).transpose(1, 2)
        m = mask.unsqueeze(-1).float()
        return (se * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * 768 + 1e-8)     # [B]

    @torch.no_grad()
    def sample(self, frame_hidden, mask=None, steps=50, cfg_w=2.0, tau=None):
        """S27: Euler-sample the normalized RESIDUAL from the FM (CFG on the score c), then
        output = μ + τ·residual (μ = det_head, NOT part of CFG). τ=0 → exactly the det_head FLOOR
        (safety net); τ=1 → full sharp. Legacy (fm_residual=False) samples cv directly. Returns raw [B,T,768]."""
        B, T, _ = frame_hidden.shape
        tau = self.tau if tau is None else tau
        x = torch.randn(B, 768, T, device=frame_hidden.device)
        no_drop = torch.zeros(B, dtype=torch.bool, device=x.device)
        do_drop = torch.ones(B, dtype=torch.bool, device=x.device)
        for k in range(steps):
            tv = k / steps
            t = torch.full((B,), tv, device=x.device)
            if cfg_w != 1.0:
                x1c = self.fm_decoder(x, t, frame_hidden, mask, no_drop)
                x1u = self.fm_decoder(x, t, frame_hidden, mask, do_drop)
                x1hat = x1u + cfg_w * (x1c - x1u)
            else:
                x1hat = self.fm_decoder(x, t, frame_hidden, mask, None)
            v = (x1hat - x) / max(1.0 - tv, 1.0 / steps)
            x = x + v * (1.0 / steps)
        if self.fm_residual:
            residual = x * self.res_std[None, :, None]                       # de-normalize the residual [B,768,T]
            mu = self.det_head(frame_hidden).transpose(1, 2)                 # [B,768,T] normalized conditional mean
            x1n = mu + tau * residual                                        # normalized cv = μ + τ·residual
        else:
            x1n = x
        return x1n.transpose(1, 2) * self.cv_std + self.cv_mean              # [B,T,768] raw

    def set_stage(self, stage):
        """S27 staged det_head+FM. stage 1 = train encoder+det_head+F0 to the FLOOR (det_loss); FM frozen;
        cond-dropout ON (de-isolate the encoder). stage 2 = FREEZE encoder/det_head/F0/decoder/pos; train ONLY
        the FM on the residual; cond-dropout OFF (deterministic gated frame_hidden = inference-consistent → the
        det_head floor is LOCKED, the FM cannot mush it). Build the optimizer on the requires_grad params AFTER this."""
        self.stage = stage
        if stage == 1:
            self.cond_dropout_on = True
            for p in self.parameters():
                p.requires_grad_(True)
            for p in self.fm_decoder.parameters():
                p.requires_grad_(False)          # FM is unused + untrained in stage 1
        else:
            self.cond_dropout_on = False          # gated, deterministic frame_hidden (matches inference)
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.fm_decoder.parameters():
                p.requires_grad_(True)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
