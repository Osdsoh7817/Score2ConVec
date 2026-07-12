"""Render the de-risk dumps (derisk_energy.py Stage C) to SoVITS wav for EAR.

Reuses the VERIFIED glue in synth_sovits.py (东雪莲 model, sid=0, cv 50->86fps, uv=(f0<30), net_g.infer).
Per clip we render: GT-cv (reference), energy-g pred-cv, mse-g pred-cv — ALL with the SAME GT-F0/uv (isolates
the cv contribution). Output: processed/derisk_wav/<clipstem>_{GT,energy_pred,mse_pred}.wav.
"""
import sys, traceback
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch, soundfile
import synth_sovits as ss

DUMPROOT = Path("processed/derisk_dump")
OUTDIR = Path("processed/derisk_wav"); OUTDIR.mkdir(parents=True, exist_ok=True)


def render_cv(net_g, dev, cv, f0):
    uv = (f0 < 30).astype(np.float32)
    cv_rs, T = ss.resample_2d(cv, ss.CV_FPS, ss.SOVITS_FPS)
    f0_rs, _ = ss.resample_1d(f0, ss.CV_FPS, ss.SOVITS_FPS)
    uv_rs, _ = ss.resample_1d(uv, ss.CV_FPS, ss.SOVITS_FPS); uv_rs = (uv_rs > 0.5).astype(np.int64)
    n = min(len(f0_rs), T)
    return ss.synthesize(net_g, dev, cv_rs[:n], np.clip(f0_rs[:n], 0, 1100), uv_rs[:n], speaker_id=0)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev); torch.cuda.synchronize()
    print("loading SoVITS (Dongxuelian) ...")
    net_g, hps = ss.load_sovits(ss.MODEL_PATH, ss.CONFIG_PATH, dev)
    gt_done = set()
    losses = sorted(p.name for p in DUMPROOT.iterdir() if p.is_dir()) if DUMPROOT.exists() else []
    for loss in losses:
        d = DUMPROOT / loss
        for npz in sorted(d.glob("*.npz")):
            z = np.load(npz); f0 = z["f0"]
            stem = npz.stem
            try:
                for key in [k for k in z.files if k.startswith("pred")]:
                    outp = OUTDIR / f"{stem}_{loss}_{key}.wav"
                    if outp.exists():            # idempotent: don't re-render finished wavs
                        continue
                    soundfile.write(str(outp), render_cv(net_g, dev, z[key], f0), ss.SOVITS_SR)
                gtp = OUTDIR / f"{stem}_GT.wav"
                if stem not in gt_done and not gtp.exists():
                    soundfile.write(str(gtp), render_cv(net_g, dev, z["gt"], f0), ss.SOVITS_SR)
                gt_done.add(stem)
                print(f"  {stem} [{loss}] ok")
            except Exception as e:
                print(f"  {stem} [{loss}] FAILED: {e}"); traceback.print_exc()
    print(f"done -> {OUTDIR}")


if __name__ == "__main__":
    main()
