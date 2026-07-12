"""Validate the 256-d ScoreToCV (sovits4.0 retarget): render 銀の龍 through the MinamiyaAkiko 4.0 model.
cv = ScoreToCV(model_cv256) [256-d], f0 = noteonly (exact notes), decode = synth_sovits via render_derisk.render_cv.
  python scripts/render_akiko.py --ust your_song.ust --model your_sovits40.pth --config config.json
"""
import sys, os, argparse
from pathlib import Path
import numpy as np, torch, yaml, soundfile
ROOT = Path(__file__).resolve().parent.parent   # repo root
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import render_ust as ru
import render_derisk as rd
import synth_sovits as ss
from src.model.score2cv import ScoreToCV

ap = argparse.ArgumentParser()
ap.add_argument("--ust", default=os.environ.get("S2CV_UST", ""), help="path to a .ust score")
ap.add_argument("--ckpt", default="checkpoints/cv256_final.pt", help="256-d ScoreToCV checkpoint")
ap.add_argument("--model", default=os.environ.get("SOVITS40_MODEL", ""), help="so-vits-svc 4.0 .pth (vec256l9, ssl_dim=256)")
ap.add_argument("--config", default=os.environ.get("SOVITS40_CONFIG", ""), help="the 4.0 model's config.json")
ap.add_argument("--out", default="processed/akiko_out", help="output dir (relative to repo root)")
ap.add_argument("--cv-spk", type=int, default=49)   # kiritan; cv is speaker-invariant so minor
A = ap.parse_args()
if not (A.ust and A.model and A.config):
    raise SystemExit("need --ust, --model (so-vits 4.0 .pth), --config (its config.json). See README > Inference.")
UST = A.ust
AKIKO_MODEL = A.model
AKIKO_CONFIG = A.config

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    _ = torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev); torch.cuda.synchronize()
cvm = ScoreToCV(yaml.safe_load(open(ROOT / "configs/model_cv256.yaml", encoding="utf-8"))).to(dev).float().eval()
sd = torch.load(ROOT / A.ckpt, map_location=dev, weights_only=False)
cvm.load_state_dict(sd["model"])
step = sd.get("global_step", "x")
print(f"cv256 ckpt step={step}  cv_dim={cvm.cv_dim}", flush=True)
net_g, hps = ss.load_sovits(AKIKO_MODEL, AKIKO_CONFIG, dev)
print(f"akiko 4.0 ssl_dim={hps.model.ssl_dim} sr={hps.data.sampling_rate}", flush=True)
assert hps.model.ssl_dim == cvm.cv_dim, "akiko ssl_dim must match cv_dim (256)"

raw, *_ = ru.ust_to_score(UST)
pid, pdur, npitch, ndur, n2p, phon = ru.build_arrays(raw)
N = len(pid)
chunks = []; start = 0; cf = 0
for i in range(N):
    cf += int(pdur[i])
    if cf > 400 and phon[i] == "SP": chunks.append((start, i + 1)); start = i + 1; cf = 0
if start < N: chunks.append((start, N))
print(f"score: {N} phones, {int(pdur.sum())} frames ({pdur.sum()/50:.1f}s), {len(chunks)} chunks", flush=True)


def mk(s, e, spk):
    M = e - s; z = lambda a: torch.tensor(a[s:e], dtype=torch.long, device=dev)[None]
    return dict(phonemes=z(pid), note_pitch=z(npitch), phone_dur=z(pdur), note_dur=z(ndur),
                note_to_phone=torch.tensor(n2p[s:e] - n2p[s], dtype=torch.long, device=dev)[None],
                speaker_id=torch.tensor([spk], device=dev), lang_id=torch.tensor([2], device=dev),
                phone_mask=torch.ones(1, M, dtype=torch.bool, device=dev), technique=torch.zeros(1, M, 7, device=dev))


parts = []
with torch.no_grad():
    for (s, e) in chunks:
        oc = cvm(**mk(s, e, A.cv_spk)); T = int(oc["frame_mask"][0].sum())
        cv = cvm.infer_cv(oc["frame_hidden"])[0, :T].float().cpu().numpy()
        note = np.repeat(npitch[s:e], pdur[s:e]).astype(np.float32)
        Tm = min(T, len(note)); cv = cv[:Tm]; note = note[:Tm]
        note_hz = np.where(note > 0, 440.0 * 2 ** ((note - 69) / 12), 0.0).astype(np.float32)
        parts.append(rd.render_cv(net_g, dev, cv, note_hz))
        print(f"  chunk [{s}:{e}] T={Tm} done", flush=True)

audio = np.concatenate(parts)
audio = (audio / (np.abs(audio).max() + 1e-9) * 0.92).astype(np.float32)
out = ROOT / A.out; out.mkdir(parents=True, exist_ok=True)
fp = out / f"render_step{step}.wav"
soundfile.write(str(fp), audio, ss.SOVITS_SR)
print(f"DONE -> {fp}  {len(audio)} ({len(audio)/ss.SOVITS_SR:.1f}s @ {ss.SOVITS_SR}Hz) "
      f"rms={np.sqrt((audio**2).mean()):.3f} nonsilent={(np.abs(audio)>1e-3).mean():.0%}", flush=True)
