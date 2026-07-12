"""Auxiliary losses for Score2ContentVec v3 (FM): F0 + voicing + pos_aux.

The contentvec target itself is modeled GENERATIVELY (flow-matching MSE) inside the
model (Score2ContentVec.fm_loss) — train.py adds it. This module keeps only the
deterministic auxiliary heads. All discrete-token machinery (CE, soft-token, codebook,
frame-consistency KL) is gone (continuous generative target; see project-token-rootcause).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Score2ContentVecLoss(nn.Module):
    """Auxiliary loss: F0 (cents, dirty-masked) + voicing BCE + pos_aux."""

    def __init__(self, config):
        super().__init__()
        tc = config["training"]
        self.f0_w = tc["f0_weight"]
        self.voicing_w = tc["voicing_weight"]
        self.sp_id = config["phoneme"]["sp_id"]
        self.pos_aux_w = tc.get("pos_aux_weight", 0.0)
        self.f0_cents_clip = tc.get("f0_cents_clip", 1200.0)         # ±octave margin beyond note window
        self.f0_smooth_l1_beta = tc.get("f0_smooth_l1_beta", 75.0)   # SmoothL1 beta (0 -> L1)

    def forward(self, pred, targets, frame_phone_ids, frame_phone_dur,
                frame_note_pitch=None, detailed_metrics=False):
        """pred: model output dict (f0/voicing/masks); targets: batch dict (f0/voicing).
        Returns (aux_total, loss_dict). The FM contentvec loss is added by train.py."""
        mask = pred["frame_mask"]
        B, T = mask.shape

        f0_pred = pred["f0"][:, :T]
        f0_target = targets["f0"][:, :T]
        voicing_target = targets["voicing"][:, :T]
        voicing_logit = pred["voicing_logit"][:, :T]

        sp_mask = frame_phone_ids[:, :T] == self.sp_id
        pitched_mask = frame_note_pitch[:, :T] > 0 if frame_note_pitch is not None else torch.ones_like(mask)

        # -- F0 loss in cents deviation space (3-note-window octave mask + jump + SmoothL1) --
        voiced_mask = mask & (f0_target > 30) & ~sp_mask & pitched_mask
        # paired_speech: note = F0-inferred median, not a real score note -> exclude from cents-F0 loss
        is_speech = targets.get("frame_is_speech") if isinstance(targets, dict) else None
        if is_speech is not None:
            voiced_mask = voiced_mask & ~is_speech[:, :T]
        f0_cents_pred = pred.get("f0_cents")
        if voiced_mask.any() and f0_cents_pred is not None:
            f0_cents_pred_v = f0_cents_pred[:, :T][voiced_mask]
            cur_midi = frame_note_pitch[:, :T].float()
            midi_hz = 440.0 * (2 ** ((cur_midi - 69) / 12))
            f0_cents_target = 1200.0 * torch.log2(
                f0_target[voiced_mask] / midi_hz[voiced_mask].clamp(min=1e-6)
            )
            # dirty-target mask: cents outside {prev,cur,next}-note window ±octave, OR octave-jump
            cur_v = cur_midi[voiced_mask]
            nlo = pred["frame_note_lo"][:, :T][voiced_mask] if "frame_note_lo" in pred else cur_v
            nhi = pred["frame_note_hi"][:, :T][voiced_mask] if "frame_note_hi" in pred else cur_v
            lo_c = (nlo - cur_v) * 100.0 - self.f0_cents_clip
            hi_c = (nhi - cur_v) * 100.0 + self.f0_cents_clip
            jump = self._octave_jump(f0_target[:, :T], mask)[voiced_mask]
            clean = (f0_cents_target >= lo_c) & (f0_cents_target <= hi_c) & ~jump
            if clean.any():
                f0_loss = F.smooth_l1_loss(
                    f0_cents_pred_v[clean], f0_cents_target[clean],
                    beta=self.f0_smooth_l1_beta,
                )
            else:
                f0_loss = torch.tensor(0.0, device=mask.device)
        elif voiced_mask.any():
            f0_loss = F.l1_loss(pred["f0"][:, :T][voiced_mask], f0_target[voiced_mask])
        else:
            f0_loss = torch.tensor(0.0, device=mask.device)

        # -- Voicing BCE --
        if mask.any():
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logit[mask], voicing_target[mask]
            )
        else:
            voicing_loss = torch.tensor(0.0, device=mask.device)

        # -- pos_aux: keep position features decodable (v2 lesson) --
        pos_pred = pred.get("pos_pred")
        pos_targets = pred.get("pos_targets")
        if self.pos_aux_w > 0 and pos_pred is not None and pos_targets is not None and mask.any():
            pos_loss = F.l1_loss(
                pos_pred[:, :T][mask], pos_targets[:, :T][mask].clamp(0, 1)
            )
        else:
            pos_loss = torch.tensor(0.0, device=mask.device)

        aux_total = (
            self.f0_w * f0_loss
            + self.voicing_w * voicing_loss
            + self.pos_aux_w * pos_loss
        )

        loss_dict = {
            "f0": f0_loss.item(),
            "voicing": voicing_loss.item(),
            "pos_aux": pos_loss.item(),
        }
        with torch.no_grad():
            if voiced_mask.any():
                f0_cents = 1200 * torch.log2(
                    (f0_pred[voiced_mask] + 1e-6) / (f0_target[voiced_mask] + 1e-6)
                ).abs()
                loss_dict["f0_mae_cents"] = f0_cents.mean().item()

        return aux_total, loss_dict

    def _octave_jump(self, f0, mask, thresh_oct=0.583, k=9):
        """[B,T] bool: voiced frames jumping > thresh_oct octaves from the LOCAL voiced-median
        log-F0 = RMVPE doubling/halving. Smooth legato/glissando stay near the median (not flagged)."""
        voiced = f0 > 30
        lf = torch.log2(f0.clamp(min=1.0))
        pad = k // 2
        win = F.pad(lf, (pad, pad)).unfold(1, k, 1)
        winv = F.pad(voiced.float(), (pad, pad)).unfold(1, k, 1)
        med = win.masked_fill(winv < 0.5, float("nan")).nanmedian(dim=-1).values
        return voiced & torch.isfinite(med) & ((lf - med).abs() > thresh_oct)
