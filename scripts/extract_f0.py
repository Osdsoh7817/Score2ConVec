"""Extract F0 (RMVPE) at 50fps from all WAV files."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import librosa
import torch
from preprocessing.rmvpe import RMVPE

DATASETS_DIR = BASE_DIR / "datasets"
ALIGNMENT_DIR = BASE_DIR / "processed" / "alignment"
OUTPUT_DIR = BASE_DIR / "processed" / "f0"
MODEL_PATH = BASE_DIR / "pretrained" / "rmvpe.pt"
TARGET_SR = 16000


def process_dataset(rmvpe_model, jsonl_path: Path, resume: bool = True) -> dict:
    name = jsonl_path.stem
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    total = len(lines)
    out_dir = OUTPUT_DIR / name

    errors: list[str] = []
    skipped = 0
    written = 0

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
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            f0_100 = rmvpe_model(audio)  # 100fps
            f0_50 = f0_100[::2]  # downsample to 50fps

            # Align to target frame count
            if len(f0_50) > target_frames:
                f0_50 = f0_50[:target_frames]
            elif len(f0_50) < target_frames:
                f0_50 = np.pad(f0_50, (0, target_frames - len(f0_50)))

            np.save(npy_path, f0_50.astype(np.float32))
            written += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors.append(f"{utt_id}: OOM")
        except Exception as e:
            errors.append(f"{utt_id}: {e}")

    return {
        "name": name, "total": total, "written": written,
        "skipped": skipped, "errors": len(errors),
        "error_details": errors[:10],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RMVPE model...")
    rmvpe_model = RMVPE(str(MODEL_PATH))
    print(f"Ready ({torch.cuda.get_device_name(0)})\n")

    jsonl_files = sorted(ALIGNMENT_DIR.glob("*.jsonl"))
    all_stats = []
    t0 = time.time()

    for jf in jsonl_files:
        print(f"[{jf.stem}]")
        t1 = time.time()
        stats = process_dataset(rmvpe_model, jf)
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

    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.npy"))
    print(f"Disk: {total_bytes / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
