"""Step 2: SoVITS synthesis from saved ContentVec + F0 features.

Usage:
  python scripts/synth_sovits.py
"""

import sys
import os
import types
import json
from pathlib import Path

# Mock faiss before any SoVITS imports
sys.modules['faiss'] = types.ModuleType('faiss')

import numpy as np
import torch
import torch.nn.functional as F
import soundfile

# Point SOVITS_ROOT at your local so-vits-svc checkout (its `utils.py` / `models.py` are imported below).
#   Windows : set SOVITS_ROOT=D:\path\to\so-vits-svc
#   Linux/mac: export SOVITS_ROOT=/path/to/so-vits-svc
SOVITS_ROOT = os.environ.get("SOVITS_ROOT")
if not SOVITS_ROOT:
    raise RuntimeError(
        "SOVITS_ROOT is not set. Point it at your local so-vits-svc checkout "
        "(the folder containing utils.py and models.py). See README > Inference.")
sys.path.insert(0, SOVITS_ROOT)

import utils as sovits_utils
from models import SynthesizerTrn

CV_FPS = 50
SOVITS_SR = 44100
SOVITS_HOP = 512
SOVITS_FPS = SOVITS_SR / SOVITS_HOP

# Your SVC voicebank (the .pth) and its config.json. Set via env or edit here.
#   set SOVITS_MODEL=...\your_model.pth  &  set SOVITS_CONFIG=...\config.json
MODEL_PATH = os.environ.get("SOVITS_MODEL", "")
CONFIG_PATH = os.environ.get("SOVITS_CONFIG", "")
INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("S2CV_INPUT_DIR", "")


def load_sovits(model_path, config_path, device):
    hps = sovits_utils.get_hparams_from_file(config_path, True)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = sovits_utils.load_checkpoint(model_path, net_g, None)
    _ = net_g.eval().to(device)
    return net_g, hps


def resample_1d(arr, src_fps, tgt_fps):
    T_tgt = int(round(len(arr) * tgt_fps / src_fps))
    t = torch.from_numpy(arr).float().view(1, 1, -1)
    out = F.interpolate(t, size=T_tgt, mode='nearest')[0, 0]
    return out.numpy(), T_tgt


def resample_2d(arr, src_fps, tgt_fps):
    T_tgt = int(round(arr.shape[0] * tgt_fps / src_fps))
    t = torch.from_numpy(arr).float().unsqueeze(0).transpose(1, 2)
    out = F.interpolate(t, size=T_tgt, mode='nearest')[0].transpose(0, 1)
    return out.numpy(), T_tgt


def synthesize(net_g, device, cv_features, f0, uv, speaker_id=0):
    c = torch.from_numpy(cv_features).float().to(device).transpose(0, 1).unsqueeze(0)
    f0_t = torch.from_numpy(f0).float().to(device).unsqueeze(0)
    uv_t = torch.from_numpy(uv).long().to(device).unsqueeze(0)
    sid = torch.LongTensor([speaker_id]).to(device).unsqueeze(0)

    with torch.no_grad():
        audio, _ = net_g.infer(c, f0=f0_t, g=sid, uv=uv_t, predict_f0=False, vol=None)
    return audio[0, 0].cpu().numpy()


def main():
    if not (MODEL_PATH and CONFIG_PATH and INPUT_DIR):
        raise RuntimeError("Set SOVITS_MODEL, SOVITS_CONFIG, and pass an input dir (argv[1] or S2CV_INPUT_DIR). "
                           "See README > Inference.")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    if device.type == "cpu":
        torch.backends.cudnn.enabled = False  # avoid LSTM flatten_parameters cuDNN probe crash on CPU-only
    print(f"Device: {device}")

    print("Loading SoVITS model...")
    net_g, hps = load_sovits(MODEL_PATH, CONFIG_PATH, device)
    print(f"  Speaker: {dict(hps.spk)}, SSL dim: {hps.model.ssl_dim}")

    input_dir = Path(INPUT_DIR)
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        name = sample_dir.name
        print(f"\n=== {name} (cos={meta.get('cos_sim', '?')}) ===")

        variants = [
            ("gt", "gt_cv.npy", "gt_f0.npy", "gt_uv.npy"),
            ("pred", "pred_cv.npy", "pred_f0.npy", "pred_uv.npy"),
            ("pred_gtf0", "pred_cv.npy", "gt_f0.npy", "gt_uv.npy"),
        ]
        for variant, cv_file, f0_file, uv_file in variants:
            cv = np.load(sample_dir / cv_file)
            f0 = np.load(sample_dir / f0_file)
            uv = np.load(sample_dir / uv_file)

            cv_rs, T = resample_2d(cv, CV_FPS, SOVITS_FPS)
            f0_rs, _ = resample_1d(f0, CV_FPS, SOVITS_FPS)
            uv_rs, _ = resample_1d(uv.astype(np.float32), CV_FPS, SOVITS_FPS)
            uv_rs = (uv_rs > 0.5).astype(np.int64)

            f0_len = min(len(f0_rs), T)
            cv_rs = cv_rs[:f0_len]
            f0_rs = f0_rs[:f0_len]
            uv_rs = uv_rs[:f0_len]

            f0_rs = np.clip(f0_rs, 0, 1100)

            try:
                audio = synthesize(net_g, device, cv_rs, f0_rs, uv_rs, speaker_id=0)
                out_path = sample_dir / f"{variant}_sovits.wav"
                soundfile.write(str(out_path), audio, SOVITS_SR)
                dur = len(audio) / SOVITS_SR
                print(f"  {variant}: {dur:.2f}s -> {out_path}")
            except Exception as e:
                print(f"  {variant}: FAILED - {e}")
                import traceback
                traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
