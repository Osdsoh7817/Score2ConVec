#!/usr/bin/env python3
"""Extract ContentVec features (768-dim @50fps) from all WAV files."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn
from transformers import HubertModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
ALIGNMENT_DIR = BASE_DIR / "processed" / "alignment"
OUTPUT_DIR = BASE_DIR / "processed" / "contentvec"
MODEL_DIR = BASE_DIR / "pretrained" / "content-vec-best"
TARGET_SR = 16000
LAYER = 12


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


@torch.no_grad()
def extract(model, wav_path: Path, target_frames: int) -> np.ndarray:
    """Load WAV, extract ContentVec layer 12, align to target frame count."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    wav = wav.cuda()
    out = model(wav, output_hidden_states=True)
    feat = out.hidden_states[LAYER].squeeze(0).cpu().numpy()  # (T, 768)

    T = feat.shape[0]
    if T > target_frames:
        feat = feat[:target_frames]
    elif T < target_frames:
        feat = np.pad(feat, ((0, target_frames - T), (0, 0)), mode="edge")

    return feat.astype(np.float16)


def process_dataset(model, jsonl_path: Path, resume: bool = True) -> dict:
    name = jsonl_path.stem
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    total = len(lines)
    out_dir = OUTPUT_DIR / name

    errors: list[str] = []
    skipped = 0
    written = 0
    frame_diffs: list[int] = []

    for i, line in enumerate(lines):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{total}...")

        rec = json.loads(line)
        utt_id = rec["utt_id"]
        target_frames = rec["total_frames"]
        npy_path = out_dir / (utt_id + ".npy")

        if resume and npy_path.exists():
            skipped += 1
            continue

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = DATASETS_DIR / rec["wav_path"]

        try:
            feat = extract(model, wav_path, target_frames)
            np.save(npy_path, feat)
            written += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                data, sr = sf.read(str(wav_path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                wav = torch.from_numpy(data).unsqueeze(0)
                if sr != TARGET_SR:
                    wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
                model_cpu = model.cpu()
                out = model_cpu(wav, output_hidden_states=True)
                feat = out.hidden_states[LAYER].squeeze(0).numpy()
                if feat.shape[0] > target_frames:
                    feat = feat[:target_frames]
                elif feat.shape[0] < target_frames:
                    feat = np.pad(feat, ((0, target_frames - feat.shape[0]), (0, 0)), mode="edge")
                np.save(npy_path, feat.astype(np.float16))
                model.cuda()
                written += 1
                errors.append(f"{utt_id}: OOM, fell back to CPU")
            except Exception as e2:
                model.cuda()
                errors.append(f"{utt_id}: OOM+CPU fail: {e2}")
        except Exception as e:
            errors.append(f"{utt_id}: {e}")

    return {
        "name": name, "total": total, "written": written,
        "skipped": skipped, "errors": len(errors),
        "error_details": errors[:10],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ContentVec model...")
    model = HubertModelWithFinalProj.from_pretrained(str(MODEL_DIR))
    model.eval()
    model.cuda()
    print(f"Ready ({torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB)\n")

    jsonl_files = sorted(ALIGNMENT_DIR.glob("*.jsonl"))
    all_stats = []
    t0 = time.time()

    for jf in jsonl_files:
        print(f"[{jf.stem}]")
        t1 = time.time()
        stats = process_dataset(model, jf)
        dt = time.time() - t1
        all_stats.append(stats)
        print(f"  {stats['written']} new + {stats['skipped']} skip "
              f"/ {stats['total']} ({dt:.1f}s) err={stats['errors']}")
        for e in stats["error_details"]:
            print(f"  ERR: {e}")
        print()

    elapsed = time.time() - t0
    total_w = sum(s["written"] for s in all_stats)
    total_s = sum(s["skipped"] for s in all_stats)
    total_t = sum(s["total"] for s in all_stats)
    total_err = sum(s["errors"] for s in all_stats)

    print("=" * 60)
    print(f"Total: {total_w} new + {total_s} skipped / {total_t} "
          f"({elapsed:.0f}s = {elapsed / 60:.1f}min)")
    print(f"Errors: {total_err}")

    # Disk usage
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.npy"))
    print(f"Disk: {total_bytes / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
