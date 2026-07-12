"""Training dataset for Score2ContentVec v3 (discrete token targets).

Reads split JSONL files + loads data on demand:
  - processed/npz/{dataset}/{seq:06d}.npz   — score + F0
  - processed/cv_tokens/{dataset}/{seq:06d}.npy — discrete token IDs
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from pathlib import Path

SP_ID = 3  # silence token (phoneme_vocab SPECIAL_TOKENS["SP"]); voicing target forced 0 here (Run6)

# Run9 (β): canonical language -> id for the language embedding (configs/model.yaml language.n_langs).
# Order is fixed; new langs append at the end (never reorder = would break trained embeddings).
LANG_TO_ID = {"zh": 0, "en": 1, "ja": 2, "de": 3, "fr": 4, "es": 5, "it": 6, "ko": 7, "ru": 8}


class S2CVDataset(Dataset):
    def __init__(self, data_root: str, split_file: str,
                 max_frames: int = 500, max_phones: int = 100,
                 lang_weights: dict = None, augment: bool = False,
                 aug_split_file: str = None, speech_split_file: str = None,
                 synth_cv_root: str = None, npz_override: dict = None, npz_root: str = None):
        self.npz_root = Path(npz_root) if npz_root else Path(data_root) / "npz"   # npz_root: e.g. processed/npz256 (sovits4.0 256-d retarget)
        # S51: per-dataset npz redirect, e.g. {"gtsinger_ja": "processed/npz_jaclean"} -> loads the
        # score from npz_override[ds]/{ds}/{seq}.npz (cleaned-score terminal test). Default {} = no change.
        self.npz_override = {k: Path(v) for k, v in (npz_override or {}).items()}
        # FM: continuous contentvec target. Real clips read npz['contentvec']; pitch-aug clips
        # read the re-extracted raw contentvec (contentvec_aug/). cv_tokens/codebook are gone.
        self.cv_aug_root = Path(data_root) / "contentvec_aug"
        # S49 mechanism test: if set, ja clips' FM target is REPLACED by the single-singer-consistent
        # DiffSinger-synth cv (processed/synth_cv/{ds}/{seq}.npy) to kill ja's one-to-many.
        self.synth_cv_root = Path(synth_cv_root) if synth_cv_root else None
        self.samples = []
        self.lang_to_idx = {}

        n_aug_missing = 0
        for sf_path in ([split_file]
                        + ([aug_split_file] if aug_split_file else [])
                        + ([speech_split_file] if speech_split_file else [])):  # Run6: multi-lang paired_speech
            if sf_path is None or not Path(sf_path).exists():
                continue
            with open(sf_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    frames = rec["frames"]
                    n_phones = rec.get("n_phones", 999)
                    if frames > max_frames or frames < 10 or n_phones > max_phones:
                        continue
                    # aug clips need their re-extracted contentvec present (robust to a
                    # partial re-extraction run); skip any that aren't ready yet.
                    sh = rec.get("aug_shift", 0)
                    if sh != 0:
                        ss = f"p{sh}" if sh > 0 else f"m{abs(sh)}"
                        if not (self.cv_aug_root / rec["dataset"] / f"{rec['seq']:06d}_{ss}.npy").exists():
                            n_aug_missing += 1
                            continue
                    # S49 mechanism test: ja keeps ONLY non-aug clips that have a synth cv (spk=42) so the
                    # ja target is FULLY single-singer-consistent; other ja (spk43/lab/aug) excluded.
                    if self.synth_cv_root is not None and rec["lang"] == "ja":
                        if sh != 0 or not (self.synth_cv_root / rec["dataset"] / f"{rec['seq']:06d}.npy").exists():
                            continue
                    self.samples.append(rec)
                    lang = rec["lang"]
                    self.lang_to_idx.setdefault(lang, []).append(len(self.samples) - 1)
        if n_aug_missing:
            print(f"[dataset] skipped {n_aug_missing} aug clips with no contentvec_aug yet")

        self.augment = augment
        self.lang_weights = lang_weights or {}
        self._build_sample_weights()

    def _build_sample_weights(self):
        self.weights = np.ones(len(self.samples), dtype=np.float64)
        for lang, indices in self.lang_to_idx.items():
            w = self.lang_weights.get(lang, 1.0)
            for idx in indices:
                self.weights[idx] = w

    def get_sampler(self):
        return WeightedRandomSampler(self.weights, len(self.samples), replacement=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        seq_str = f"{info['seq']:06d}"
        ds = info["dataset"]

        npz_base = self.npz_override.get(ds, self.npz_root)
        data = np.load(str(npz_base / ds / f"{seq_str}.npz"), allow_pickle=True)

        aug_shift = info.get("aug_shift", 0)
        if aug_shift != 0:
            shift_str = f"p{aug_shift}" if aug_shift > 0 else f"m{abs(aug_shift)}"
            contentvec = np.load(str(self.cv_aug_root / ds / f"{seq_str}_{shift_str}.npy"))
        else:
            contentvec = data["contentvec"]
            # S49 mechanism test: swap ja's FM target -> single-singer-consistent DiffSinger-synth cv
            if self.synth_cv_root is not None and info["lang"] == "ja":
                sp = self.synth_cv_root / ds / f"{seq_str}.npy"
                if sp.exists():
                    sv = np.load(str(sp))
                    if len(sv) == len(contentvec):        # only swap when frame-aligned (safety)
                        contentvec = sv
        contentvec = np.ascontiguousarray(contentvec, dtype=np.float16)   # keep fp16 (data-loading is the bottleneck)

        # per-phone technique (Run6): [n_phones, 7] multi-hot; absent (non-GTSinger) -> zeros = normal
        if "technique" in data.files:
            technique = data["technique"].astype(np.float32)
        else:
            technique = np.zeros((data["phonemes"].shape[0], 7), dtype=np.float32)
        # S19: collapse content-INERT techniques (mix/falsetto/breathy/pharyngeal = dims 0-3) into
        # 'normal'. ContentVec is content-pure -> these carry ~nothing in the target (cos normal↔tech
        # 0.97), so live flags would just dilute scarce data with un-learnable dims. Keep glissando(4)+
        # vibrato(5) (F0-side, the F0 head shapes these) + paired_speech(6) (functional speech marker).
        technique[:, 0:4] = 0.0

        note_pitch = data["note_pitch"].astype(np.int64)
        f0 = data["f0"].astype(np.float32)
        speaker_id = info["speaker_id"]

        if aug_shift != 0:
            voiced = note_pitch > 0
            note_pitch[voiced] = np.clip(note_pitch[voiced] + aug_shift, 1, 127)
            f0_voiced = f0 > 30
            f0[f0_voiced] = f0[f0_voiced] * (2 ** (aug_shift / 12))

        if self.augment:
            # S29: the id-75 "generic singer" anon trick was REMOVED — the S27
            # conditioning_dropout (spk_dropout=0.5 zeros the embed in forward)
            # supersedes it, and id 75 is now a REAL speaker (gt_ZH-Alto-1).
            if aug_shift == 0 and random.random() < 0.3:
                shift = random.choice([-2, -1, 1, 2])
                voiced = note_pitch > 0
                note_pitch[voiced] = np.clip(note_pitch[voiced] + shift, 1, 127)
                f0_voiced = f0 > 30
                f0[f0_voiced] = f0[f0_voiced] * (2 ** (shift / 12))

        return {
            "phonemes": torch.from_numpy(data["phonemes"].astype(np.int64)),
            "note_pitch": torch.from_numpy(note_pitch),
            "phone_dur": torch.from_numpy(data["phone_dur"].astype(np.int64)),
            "note_dur": torch.from_numpy(data["note_dur"].astype(np.int64)),
            "note_to_phone": torch.from_numpy(data["note_to_phone"].astype(np.int64)),
            "contentvec": torch.from_numpy(contentvec),   # [T,768] float32 FM target
            "f0": torch.from_numpy(f0),
            "speaker_id": speaker_id,
            "technique": torch.from_numpy(technique),
            "lang": info["lang"],
            # pre-computed pitch-aug (shifted contentvec). Its UNVOICED frames (stops/silence) are
            # phase-vocoder artifact-prone -> masked from FM loss. On-the-fly aug (aug_shift==0 +
            # random note shift) keeps REAL contentvec, so it's NOT flagged.
            "is_aug": bool(aug_shift != 0),
        }

    def stats(self):
        lang_counts = {lang: len(idxs) for lang, idxs in self.lang_to_idx.items()}
        total_frames = sum(s["frames"] for s in self.samples)
        return {
            "total_samples": len(self.samples),
            "total_frames": total_frames,
            "total_hours": total_frames / 50 / 3600,
            "per_lang": lang_counts,
        }


def collate_fn(batch):
    """Pad to fixed shapes for consistent cuDNN performance."""
    # dynamic pad to the batch max (backward-compat: never below the old fixed 100/500,
    # so ≤500-frame runs are byte-identical; only batches with longer clips expand).
    max_phones = max(100, max(int(b["phonemes"].shape[0]) for b in batch))
    max_frames = max(500, max(int(b["contentvec"].shape[0]) for b in batch))

    B = len(batch)
    result = {
        "phonemes": torch.zeros(B, max_phones, dtype=torch.long),
        "note_pitch": torch.zeros(B, max_phones, dtype=torch.long),
        "phone_dur": torch.zeros(B, max_phones, dtype=torch.long),
        "note_dur": torch.zeros(B, max_phones, dtype=torch.long),
        "note_to_phone": torch.zeros(B, max_phones, dtype=torch.long),
        "technique": torch.zeros(B, max_phones, 7),
        "phone_mask": torch.zeros(B, max_phones, dtype=torch.bool),
        "speaker_id": torch.zeros(B, dtype=torch.long),
        "contentvec": torch.zeros(B, max_frames, batch[0]["contentvec"].shape[1], dtype=torch.float16),   # FM target (fp16); dim-aware: 768 or 256
        "f0": torch.zeros(B, max_frames),
        "voicing": torch.zeros(B, max_frames),
        "frame_phone_ids": torch.zeros(B, max_frames, dtype=torch.long),
        "frame_phone_dur": torch.zeros(B, max_frames, dtype=torch.long),
        "frame_note_pitch": torch.zeros(B, max_frames, dtype=torch.long),
        "frame_has_tech": torch.zeros(B, max_frames, dtype=torch.bool),  # Run6: any technique flag at this frame
        "frame_is_speech": torch.zeros(B, max_frames, dtype=torch.bool),  # Run6: paired_speech (F0-vs-note rule differs)
    }

    for i, b in enumerate(batch):
        N = b["phonemes"].shape[0]
        T = b["contentvec"].shape[0]

        result["phonemes"][i, :N] = b["phonemes"]
        result["note_pitch"][i, :N] = b["note_pitch"]
        result["phone_dur"][i, :N] = b["phone_dur"]
        result["note_dur"][i, :N] = b["note_dur"]
        result["note_to_phone"][i, :N] = b["note_to_phone"]
        result["technique"][i, :N] = b["technique"]
        result["phone_mask"][i, :N] = True
        result["speaker_id"][i] = b["speaker_id"]
        result["contentvec"][i, :T] = b["contentvec"]
        result["f0"][i, :T] = b["f0"]
        result["voicing"][i, :T] = (b["f0"] > 30).float()

        phone_ids_expanded = torch.repeat_interleave(b["phonemes"], b["phone_dur"])
        phone_dur_expanded = torch.repeat_interleave(b["phone_dur"], b["phone_dur"])
        note_pitch_expanded = torch.repeat_interleave(b["note_pitch"], b["phone_dur"])
        t = min(T, phone_ids_expanded.shape[0])
        result["frame_phone_ids"][i, :t] = phone_ids_expanded[:t]
        result["frame_phone_dur"][i, :t] = phone_dur_expanded[:t]
        result["frame_note_pitch"][i, :t] = note_pitch_expanded[:t]
        # Run6: force voicing target = 0 at SP frames (teach SP=unvoiced/silence; "浊帧去除")
        sp_fr = phone_ids_expanded[:t] == SP_ID
        result["voicing"][i, :t] = result["voicing"][i, :t].masked_fill(sp_fr, 0.0)
        result["frame_has_tech"][i, :t] = torch.repeat_interleave(
            b["technique"].any(dim=-1), b["phone_dur"])[:t]
        result["frame_is_speech"][i, :t] = torch.repeat_interleave(
            b["technique"][:, 6].bool(), b["phone_dur"])[:t]

    result["lang"] = [b["lang"] for b in batch]
    result["lang_id"] = torch.tensor([LANG_TO_ID.get(b["lang"], 0) for b in batch], dtype=torch.long)
    result["is_aug"] = torch.tensor([b["is_aug"] for b in batch], dtype=torch.bool)
    return result
