#!/usr/bin/env python3
"""Extract vec256l9 ContentVec (256-dim @50fps) = final_proj(layer 9) — for so-vits-svc 4.0 retarget.

Mirrors extract_contentvec.py EXACTLY (same model dir, same audio load, same target_frames alignment,
same float16, same per-utt output) but: LAYER=9 + apply final_proj -> 256-dim, output -> contentvec256/.
Verified bit-identical to so-vits's fairseq ContentVec256L9 (cos=1.0). Only the 12 final-lineup datasets.
"""
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
OUTPUT_DIR = BASE_DIR / "processed" / "contentvec256"
MODEL_DIR = BASE_DIR / "pretrained" / "content-vec-best"
TARGET_SR = 16000
LAYER = 9  # vec256l9: layer 9 + final_proj

LINEUP = {"gtsinger_en", "itako", "kiritan", "m4singer", "mfa_french", "mfa_german",
          "mfa_italian", "mfa_spanish", "natsume", "ofuton", "oniku", "pjs"}


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


@torch.no_grad()
def extract(model, wav_path: Path, target_frames: int) -> np.ndarray:
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.cuda()
    out = model(wav, output_hidden_states=True)
    feat = model.final_proj(out.hidden_states[LAYER]).squeeze(0).cpu().numpy()  # (T, 256)

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
    errors, skipped, written = [], 0, 0

    for i, line in enumerate(lines):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{total}...", flush=True)
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
                feat = model_cpu.final_proj(out.hidden_states[LAYER]).squeeze(0).numpy()
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

    return {"name": name, "total": total, "written": written, "skipped": skipped,
            "errors": len(errors), "error_details": errors[:10]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading ContentVec (final_proj) model...")
    model = HubertModelWithFinalProj.from_pretrained(str(MODEL_DIR))
    model.eval().cuda()
    print(f"Ready ({torch.cuda.get_device_name(0)})  LAYER={LAYER} -> 256-d\n")

    jsonl_files = sorted(f for f in ALIGNMENT_DIR.glob("*.jsonl") if f.stem in LINEUP)
    print(f"Datasets ({len(jsonl_files)}): {[f.stem for f in jsonl_files]}\n")
    t0 = time.time()
    all_stats = []
    for jf in jsonl_files:
        print(f"[{jf.stem}]", flush=True)
        t1 = time.time()
        stats = process_dataset(model, jf)
        all_stats.append(stats)
        print(f"  {stats['written']} new + {stats['skipped']} skip / {stats['total']} "
              f"({time.time()-t1:.1f}s) err={stats['errors']}", flush=True)
        for e in stats["error_details"]:
            print(f"  ERR: {e}")

    elapsed = time.time() - t0
    tw = sum(s["written"] for s in all_stats)
    ts = sum(s["skipped"] for s in all_stats)
    terr = sum(s["errors"] for s in all_stats)
    tb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.npy"))
    print(f"\n{'='*50}\nTotal: {tw} new + {ts} skipped ({elapsed/60:.1f}min)  Errors: {terr}")
    print(f"Disk: {tb/1024**3:.1f} GB -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
