"""ScoreToCV (S58 FINAL) — the clean, elegant score→ContentVec model.

The deterministic-head cv predictor, stripped of all the dead weight the research model carried:
  REMOVED: FlowMatchingDecoder/DiTBlock (FM never deployed — tau=0 = det_head), F0Predictor (F0 is now a
           SEPARATE FM model), f0_condition/encoder_f0 (flag-gated dead paths), residual-norm, CFG, stages.
  KEPT (the proven core): embeds + cond-dropout + ConformerEncoder + soft LengthRegulator + 1-layer
           ConformerBlock decoder + DetHead + pos_aux. Layer names match score2contentvec.py so a trained
           checkpoint loads weight-for-weight (extract-verify).
Deploy: infer_cv(frame_hidden) = det_head mean, de-normalized. F0 comes from the separate f0 model.
~39M params (vs ~72M with the dead FM). Trains on det_loss + pos_aux only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from torch.utils.checkpoint import checkpoint

from .modules import ConformerEncoder, ConformerBlock, RelativePositionalEncoding
from .score2contentvec import LengthRegulator, DetHead   # reuse the proven sub-modules (same internals)


class ScoreToCV(nn.Module):
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

        # per-phone technique (Run6 de-averaging): zero-init -> no-op; absorbs gtsinger expression deltas so
        # the normal-cv stays clean. dims 0-3 zeroed upstream (dataset.py); deploy feeds zeros = no-op.
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

        # frame-level decoder (denoise alignment jitter / boundary-bleed)
        dec = config.get("decoder", {})
        self.dec_layers = dec.get("num_layers", 0)
        self.dec_checkpoint = dec.get("grad_checkpoint", False)
        if self.dec_layers > 0:
            self.dec_pos_enc = RelativePositionalEncoding(ec["dim"])
            self.decoder = nn.ModuleList([
                ConformerBlock(ec["dim"], dec.get("num_heads", ec["num_heads"]), dec.get("ff_mult", 4),
                               dec.get("conv_kernel", 15), dec.get("dropout", ec["dropout"]),
                               dec.get("drop_path", 0.0), use_conv=dec.get("use_conv", False))
                for _ in range(self.dec_layers)])
        else:
            self.decoder = None

        self.cv_dim = config.get("cv_dim", 768)          # 768 = ContentVec vec768l12 (default); 256 = vec256l9 (sovits4.0)
        self.det_head = DetHead(cond_dim=ec["dim"], out=self.cv_dim)

        # pos_aux head (v2: pos features go dead without this auxiliary loss)
        use_pos_aux = config["training"].get("pos_aux_weight", 0.0) > 0 and lrc["pos_features"]
        self.pos_head = nn.Linear(ec["dim"], 2) if use_pos_aux else None

        # per-dim contentvec normalization (TRAIN-only stats)
        norm_path = Path(config["fm_decoder"].get("norm_path", "processed/contentvec_norm.npz")) \
            if "fm_decoder" in config else Path(config.get("cv_norm_path", "processed/contentvec_norm.npz"))
        if norm_path.exists():
            nz = np.load(norm_path)
            self.register_buffer("cv_mean", torch.tensor(nz["mean"], dtype=torch.float32))
            self.register_buffer("cv_std", torch.tensor(nz["std"], dtype=torch.float32))
        else:
            print(f"[WARN] {norm_path} missing -> cv NOT normalized (run compute_cv_norm.py)")
            self.register_buffer("cv_mean", torch.zeros(self.cv_dim))
            self.register_buffer("cv_std", torch.ones(self.cv_dim))

        # S27 de-isolate: train-time dropout of speaker/lang embeds (pools data-poor langs)
        cdc = config.get("conditioning_dropout", {})
        self.spk_dropout = cdc.get("speaker", 0.0)
        self.lang_dropout = cdc.get("lang", 0.0)
        self.cond_dropout_on = True

    def forward(self, phonemes, note_pitch, phone_dur, note_dur, note_to_phone,
                speaker_id, lang_id=None, phone_mask=None, technique=None, f0=None):
        """f0 accepted but IGNORED (signature-compatible with the old model / dataset collate)."""
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
                keep = (torch.rand(B, device=phonemes.device) >= self.spk_dropout).float()[:, None, None]
                spk_emb = spk_emb * keep
            if self.lang_dropout > 0:
                keep = (torch.rand(B, device=phonemes.device) >= self.lang_dropout).float()[:, None, None]
                lang_emb = lang_emb * keep
        x = self.input_proj(torch.cat([phone_emb, pitch_emb, spk_emb, lang_emb], dim=-1))
        x = self.encoder(x, phone_mask)

        frame_hidden, frame_mask, phone_idx, pos_targets = self.length_reg(
            x, phone_dur, note_dur, note_to_phone)
        pos_pred = torch.sigmoid(self.pos_head(frame_hidden)) if self.pos_head is not None else None

        if self.decoder is not None:
            h = self.dec_pos_enc(frame_hidden)
            for layer in self.decoder:
                if self.dec_checkpoint and self.training:
                    h = checkpoint(layer, h, frame_mask, use_reentrant=False)
                else:
                    h = layer(h, frame_mask)
            frame_hidden = h

        return {"frame_hidden": frame_hidden, "frame_mask": frame_mask,
                "pos_pred": pos_pred, "pos_targets": pos_targets}

    def det_loss(self, frame_hidden, target_cv, mask):
        """The ONLY cv objective: det_head L2 to normalized cv = conditional mean E[cv|c]."""
        B, T, _ = frame_hidden.shape
        x1n = (target_cv[:, :T] - self.cv_mean) / self.cv_std
        mu = self.det_head(frame_hidden)
        se = (mu - x1n) ** 2
        m = mask.unsqueeze(-1).float()
        return (se * m).sum() / (m.sum() * self.cv_dim + 1e-8)

    @torch.no_grad()
    def infer_cv(self, frame_hidden):
        """Deploy output: det_head mean, de-normalized -> raw cv [B,T,768]."""
        return self.det_head(frame_hidden) * self.cv_std + self.cv_mean

    @torch.no_grad()
    def sample(self, frame_hidden, mask=None, **kwargs):
        """Render-compatible alias (ignores steps/cfg_w/tau — det model has no sampling)."""
        return self.infer_cv(frame_hidden)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
