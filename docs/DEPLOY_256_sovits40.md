# ScoreToCV-256 — so-vits-svc 4.0 Integration Guide

The **256-dim (vec256l9)** variant of ScoreToCV, for **so-vits-svc 4.0** backends. Companion to the 768
guide. Built + EAR-validated S60 (2026-06-25).

> **Open-source release note.** Written during development. `runs/cv256/checkpoints/cv256_final.pt` and
> `processed/contentvec256_norm.npz` refer to the original training layout; in this repository the released
> checkpoint is **`checkpoints/cv256_final.pt`** (norm baked in). `D:\MyDev\...` paths are the author's dev
> machine — substitute your own so-vits-svc 4.0 model.

> **The model, the input/output contract, the G2P/frontend, the conventions, and the architecture are
> IDENTICAL to the 768 model — read `runs/cvfinal/DEPLOY.md` for all of that.** This doc only covers what
> differs for the 256 / so-vits-4.0 path. Same `ScoreToCV` class, same score arrays, same 50 fps, same
> deterministic head — only the **output feature** changes (768 → 256) and the **backend** is 4.0 not 4.1.

---

## 0. What's different vs the 768 ship

| | 768 (cv_final) | **256 (cv256_final)** |
|---|---|---|
| output dim | 768 | **256** |
| target feature | ContentVec **vec768l12** (layer 12) | ContentVec **vec256l9** (layer 9 + `final_proj`) |
| backend | so-vits-svc **4.1**, RVC v2 | so-vits-svc **4.0** |
| checkpoint | `runs/cvfinal/checkpoints/cv_final.pt` | **`runs/cv256/checkpoints/cv256_final.pt`** |
| config | `configs/model_cv_final.yaml` | **`configs/model_cv256.yaml`** |
| norm stats | `processed/contentvec_norm.npz` | **`processed/contentvec256_norm.npz`** |
| render reference | `scripts/render_ust.py` / `render_derisk.render_cv` | **`scripts/render_akiko.py`** |
| params | 41,221,700 | 40,959,044 |
| det_floor | ~0.795 | **0.791** (en .803 / zh .800 / ja .791 lead; fr/de ~0.76) |

Everything else is the same model. `cv_dim` in `model_cv256.yaml` = 256; the `DetHead` output, the det-loss
normalizer, and the baked norm buffers all follow `cv_dim` automatically (one config, backward-compatible).

---

## 1. The target feature = vec256l9 (the only thing that had to change)

so-vits-svc 4.0's standard speech encoder is **ContentVec vec256l9** = the ContentVec network's **layer-9
hidden state passed through its `final_proj` (768→256)**. (A 4.0 `config.json` with `ssl_dim: 256` and **no**
`speech_encoder` field = original 4.0 = vec256l9.)

- **Extractor**: `scripts/extract_contentvec256.py` — HuggingFace `HubertModelWithFinalProj`,
  `final_proj(hidden_states[9])` → 256-d @ 50 fps (16 kHz, hop 320). **Verified bit-identical** (per-frame
  cos = 1.0000) to so-vits-svc's own fairseq `ContentVec256L9` (`pretrain/checkpoint_best_legacy_500.pt`), so
  the HF route is a guaranteed match without the fairseq dependency.
- This is the ONE feature that differs from the 768 (vec768l12) path; the alignment, durations, notes, f0, and
  data lineup are all unchanged.

---

## 2. Inference (identical to the 768 flow, with the 256 config + checkpoint)

```python
cfg = yaml.safe_load(open("configs/model_cv256.yaml", encoding="utf-8"))
cvm = ScoreToCV(cfg).to(dev).float().eval()
cvm.load_state_dict(torch.load("runs/cv256/checkpoints/cv256_final.pt", weights_only=False)["model"])
# build the SAME score arrays as the 768 guide (phonemes/note_pitch/phone_dur/note_dur/note_to_phone/
# speaker_id/lang_id/phone_mask/technique=zeros); chunk at SP ≤400 frames.
out = cvm(**inputs)
cv = cvm.infer_cv(out["frame_hidden"])[0, :T].cpu().numpy()   # [T, 256]   (was [T,768])
```
The score arrays, G2P, language ids, technique=zeros, chunking — **all exactly as in
`runs/cvfinal/DEPLOY.md` §2–§4**. The only change is the config/checkpoint and the output dim.

---

## 3. Backend: so-vits-svc 4.0 decode

Reference: **`scripts/render_akiko.py`** (renders 銀の龍 through the akiko 4.0 model). It reuses the verified
`render_derisk.render_cv` glue:
- resample cv 50 → **86.13 fps** (44.1 kHz / hop 512), `uv = (f0 < 30)`, clamp f0 ∈ [0, 1100];
- `net_g.infer(c=[1,256,T], f0, g=speaker_id, uv, predict_f0=False, vol=None)` → audio @ 44.1 kHz.
- **speaker_id** = the 4.0 model's own speaker index (read `config.json["spk"]`; for the akiko model it's
  `{"akiko4.0": 0}` ⇒ `speaker_id=0`). cv is speaker-invariant, but the *backend's* `g` must be a valid speaker
  of THAT model.
- Load: `ss.load_sovits(MODEL.pth, CONFIG.json, dev)` (the `D:\MyDev\so-vits-svc` codebase loads 4.0 and 4.1
  with the same `SynthesizerTrn` / `net_g.infer` signature; point MODEL/CONFIG at the 4.0 model).

To validate a NEW 4.0 model: `py -3.10 scripts/render_akiko.py --ckpt runs/cv256/checkpoints/cv256_final.pt`
(swap the `AKIKO_MODEL`/`AKIKO_CONFIG` paths inside to the new model; assert `hps.model.ssl_dim == 256`).

### ⚠ About the validation backend (akiko 4.0)
The so-vits-svc 4.0 model used for validation (`MinamiyaAkiko-Sovits4.0`, 320k steps, unknown provenance) is
**low-quality**: ja-biased, poor cross-lingual generalization. It garbles non-ja 咬字, drops voice (失声), and
breaks on high notes — **even when fed GROUND-TRUTH vec256l9 features** (proven by the S60 isolation diag:
`processed/s60_akiko_diag/*_gtcv*.wav` show the same artifacts on perfect features). **Those artifacts are the
akiko model, NOT our cv.** Our cv was verified clean (right feature, right speaker id, correct value scale,
correct render path). ⇒ **swap in any decent vec256l9 so-vits-4.0 voicebank and the output improves**; akiko
was only the validation vehicle.

---

## 4. Re-build recipe (if you ever re-extract the 256 data)

The S60 pipeline (no re-align, no data/lineup change — only the target feature):
1. `scripts/extract_contentvec256.py` → `processed/contentvec256/` (per-utt 256-d @50fps).
2. `scripts/build_npz256.py` → `processed/npz256/` = **reuse each existing npz's fields (phonemes/dur/notes/f0)
   verbatim, swap ONLY the cv** to the frame-aligned 256 slice. ⚠ **Do NOT rebuild from
   `manifest_final.jsonl`** — it is STALE (predates the S46 dict-patch + S57 de-cram). Gate: every clip
   `cv.T == f0.T == Σphone_dur`, dim 256 (`build_npz256` reports `0 FAILS`).
3. `scripts/compute_cv_norm256.py` → `processed/contentvec256_norm.npz`.
4. `scripts/train_cv.py --config configs/model_cv256.yaml --run-name cv256` (~60k, ~2 h on a 3080 Ti).
5. Ship: extract the best/final checkpoint's `model` state into `cv256_final.pt`.
6. **Validate by EAR** through the 4.0 backend (`render_akiko.py`) — a det_floor number is not a ship gate.

---

## 5. Status

- **SHIPPED** `runs/cv256/checkpoints/cv256_final.pt` (det_floor 0.791, step 60000, 256-d, S60).
- **EAR-validated** (S60): 銀の龍 through akiko 4.0 + 256 cv + noteonly f0 = intelligible + in-tune; residual
  artifacts traced to the akiko backend (above), not our pipeline.
- With this, the multi-backend set is complete: **cv_final (768)** → so-vits-svc 4.1 + RVC v2;
  **cv256_final (256)** → so-vits-svc 4.0. f0 stays parametric-in-DAW (learned f0 retired, S59).
```
