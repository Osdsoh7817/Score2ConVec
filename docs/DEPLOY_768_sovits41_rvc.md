# ScoreToCV — Integration & Deployment Guide

**The 自己唱 content model.** Turns a musical score (notes + lyrics) into **ContentVec** features, which
any ContentVec-based SVC backend (so-vits-svc, RVC) decodes into singing. Self-contained — you should not
need to read the session memory or the training code to integrate this into a DAW/frontend.

Last updated: S60, 2026-06-25.

> **Open-source release note.** This guide was written during development. Paths such as
> `runs/cvfinal/checkpoints/cv_final.pt` and `processed/contentvec_norm.npz` refer to the original training
> layout; in this repository the released checkpoint lives at **`checkpoints/cv_final.pt`** (its norm stats are
> baked in). Any `D:\MyDev\...` paths are the author's dev machine — substitute your own so-vits-svc / RVC
> checkout and voice model.

---

## 0. TL;DR

```
DAW score ──G2P+arrays──▶ ScoreToCV ──▶ ContentVec [T, D] @50fps ──┐
(notes+lyrics)                          (det conditional mean)     ├─▶ SVC backend ─▶ singing .wav
                                                                   │   (so-vits / RVC)
f0 stream (note pitch, DAW-side) ──────────────────────────────────┘
```

- **What ScoreToCV produces** = ContentVec = *speaker-invariant CONTENT* (the "what is being sung":
  phonetic identity over time). It does **NOT** carry timbre or pitch.
- **The voice (timbre + expression)** comes from the **SVC backend**, not from us. Same cv → different
  backend = different singer. This decoupling is the whole design (low data, backend-agnostic).
- **Pitch (f0)** is a **SEPARATE stream** you supply alongside cv. cv ⊥ f0. See §7.
- **Deterministic**: same score → same cv, every time. No sampling, no seed.

---

## 1. Artifacts & key files

| Thing | Path |
|---|---|
| **Checkpoint (768-d, SHIP)** | `runs/cvfinal/checkpoints/cv_final.pt` |
| Config (768) | `configs/model_cv_final.yaml` |
| Model code | `src/model/score2cv.py` (class `ScoreToCV`) |
| IPA vocabulary (source of truth) | `src/preprocessing/phoneme_vocab.py` (`PHONE_TO_ID`, 210 tokens) |
| **Frontend reference (end-to-end)** | `scripts/render_ust.py` (UST → arrays → cv → audio) |
| SVC decode glue | `scripts/render_derisk.py` (`render_cv`) + `scripts/synth_sovits.py` |
| 768-d normalization stats | `processed/contentvec_norm.npz` (baked into the model at load) |

**Target-feature variants** (the model can output different ContentVec flavors — pick by backend):
- **768-d = vec768l12** (ContentVec layer-12) → **so-vits-svc 4.1** and **RVC v2**. This is `cv_final.pt`. SHIPPED.
- **256-d = vec256l9** (ContentVec layer-9 + final_proj) → **so-vits-svc 4.0**. **SHIPPED (S60)**:
  `runs/cv256/checkpoints/cv256_final.pt`, config `configs/model_cv256.yaml`. Identical model, output dim 256,
  det_floor 0.791. **→ see `runs/cv256/DEPLOY.md`** for the 4.0-specific integration guide.

---

## 2. Model I/O contract  ← the important part

### Inputs (all created by the frontend; deploy uses batch size B=1)

`N` = number of phones in the (chunk of) score. All are per-phone arrays of length N except the two scalars.

| name | tensor | dtype | meaning |
|---|---|---|---|
| `phonemes` | `[B, N]` | long | IPA token ids (`PHONE_TO_ID`; see §3). `SP`=3 for rests. |
| `note_pitch` | `[B, N]` | long | MIDI note per phone, 0–127. **0 = rest/SP** (unpitched). |
| `phone_dur` | `[B, N]` | long | duration of each phone in **frames** (1 frame = 20 ms; 50 fps). ≥1. |
| `note_dur` | `[B, N]` | long | total frames of the *note group* this phone belongs to (per-phone, repeated). |
| `note_to_phone` | `[B, N]` | long | note-group index per phone: 0-based, non-decreasing. **Rebase to start at 0 within each chunk.** |
| `speaker_id` | `[B]` | long | 0–76. cv is speaker-invariant ⇒ minor effect; default **49** (kiritan). |
| `lang_id` | `[B]` | long | language (see §3.2). e.g. ja=2. **Matters** (selects phonotactics). |
| `phone_mask` | `[B, N]` | bool | True = real phone (all True at deploy; used for batch padding). |
| `technique` | `[B, N, 7]` | float | **ALL ZEROS at deploy.** (see §6 gotchas) |
| `f0` | — | — | accepted by `forward(...)` but **IGNORED** (signature compat only). |

### Output

```python
out = model(**inputs)                      # dict
#   out["frame_hidden"] : [B, T, 512]  frame-level hidden (T = sum(phone_dur))
#   out["frame_mask"]   : [B, T]  bool
cv = model.infer_cv(out["frame_hidden"])   # [B, T, D]  ContentVec, DE-normalized (raw cv space)
#   D = 768 (cv_final) or 256 (cv256). T = total frames = sum(phone_dur). 50 fps.
```

`infer_cv` already de-normalizes (multiplies by the baked std + mean). **Do not normalize again** — feed
`cv` straight to the backend. The model is `@torch.no_grad()` inside `infer_cv`; run the whole thing under
`torch.no_grad()` and `.eval()`.

---

## 3. The frontend: score → arrays

The reference implementation is **`scripts/render_ust.py`** — read `ust_to_score()` + `build_arrays()`.
A DAW frontend reimplements this from its own note/lyric model. The pieces:

### 3.1 Lyrics → IPA phones (G2P)
Each note's lyric becomes 1–3 IPA phones (consonant(s) + vowel). The phones must be **IPA tokens that exist
in `phoneme_vocab.PHONE_TO_ID`** (210-token unified inventory across 9 langs; unknown → PAD=0 = silently wrong,
so validate). The vocab module ships converters from common formats:
- **ZH**: opencpop/pinyin → IPA (`OPENCPOP_INITIALS/FINALS`).
- **EN**: ARPABET/CMUdict → IPA (`ARPABET_TO_IPA`, with the AH0/AH1 schwa-vs-STRUT split).
- **JA**: kana/romaji → IPA (`JA_ROMAJI_TO_IPA`; render_ust.py has a full hand-built kana table).
- **de/fr/es/it**: MFA IPA → normalized IPA (`convert_mfa`).
Special tokens: `SP`=3 (rest/silence), `AP`=4 (breath). PAD=0, BOS=1, EOS=2 (BOS/EOS unused at deploy).

### 3.2 Language id (`lang_id`) — `src/training/dataset.py:LANG_TO_ID`
```
zh=0  en=1  ja=2  de=3  fr=4  es=5  it=6   (ko=7, ru=8 exist but were DROPPED — don't use)
```
Set `lang_id` to the song's language. The 7 shipped languages are 0–6.

### 3.3 Building the per-phone arrays (`build_arrays`)
- **phone_dur**: split each note's frame length across its phones. render_ust uses `split_dur`: leading
  consonants get ~≤4 frames each, the vowel gets the remainder. This is a heuristic — a DAW can expose it.
- **note grouping → note_to_phone / note_dur**: consecutive phones on the **same MIDI pitch** form one note
  group; `note_to_phone` = that group's running index; `note_dur` = sum of the group's `phone_dur` (repeated
  onto each phone). A held/sustained syllable ("-" / "ー") repeats the previous vowel at the new pitch.
- **rests**: emit an `SP` phone with `note_pitch=0` and some frame length (render_ust caps lead/trailing rests
  to ~25 fr, mid-song rests to ~70 fr).
- **frames**: `frames = round(ticks · 60 / tempo / 480 · 50)` for a UST (480 ticks/quarter, 50 fps).

### 3.4 Chunking (for long songs)
Split the score at `SP` boundaries into chunks of **≤ ~400 frames** (avoids SVC OOM and O(N²) attention).
**Rebase `note_to_phone` to start at 0 in each chunk** (`n2p[s:e] - n2p[s]`). Render each chunk, concatenate
the audio. A 240 s song ≈ 12 000 frames ≈ 15 chunks.

---

## 4. Minimal inference example

```python
import sys, torch, yaml, numpy as np
from pathlib import Path
sys.path.insert(0, "scripts")          # render_derisk/synth_sovits expect scripts/ on the path
from src.model.score2cv import ScoreToCV
import render_derisk as rd, synth_sovits as ss

ROOT = Path(".")
dev = torch.device("cuda")

# --- load cv model (768) ---
cfg = yaml.safe_load(open(ROOT / "configs/model_cv_final.yaml", encoding="utf-8"))
cvm = ScoreToCV(cfg).to(dev).float().eval()
cvm.load_state_dict(torch.load(ROOT / "runs/cvfinal/checkpoints/cv_final.pt",
                               map_location=dev, weights_only=False)["model"])

# --- build score arrays (from your frontend); here one chunk, lang=ja, spk=49 ---
# pid, npitch, pdur, ndur, n2p : 1-D int arrays of length N (see §3 / render_ust.build_arrays)
def to_inputs(pid, npitch, pdur, ndur, n2p, spk=49, lang=2):
    M = len(pid); L = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)[None]
    return dict(phonemes=L(pid), note_pitch=L(npitch), phone_dur=L(pdur), note_dur=L(ndur),
                note_to_phone=L(np.asarray(n2p) - n2p[0]),
                speaker_id=torch.tensor([spk], device=dev), lang_id=torch.tensor([lang], device=dev),
                phone_mask=torch.ones(1, M, dtype=torch.bool, device=dev),
                technique=torch.zeros(1, M, 7, device=dev))

with torch.no_grad():
    out = cvm(**to_inputs(pid, npitch, pdur, ndur, n2p))
    T = int(out["frame_mask"][0].sum())
    cv = cvm.infer_cv(out["frame_hidden"])[0, :T].float().cpu().numpy()   # [T, 768]

# --- f0 stream @50fps: noteonly (exact note Hz) — see §7 ---
note = np.repeat(npitch, pdur)[:T].astype(np.float32)
f0 = np.where(note > 0, 440.0 * 2 ** ((note - 69) / 12), 0.0).astype(np.float32)

# --- decode through a backend (so-vits 4.1 东雪莲 here) ---
net_g, hps = ss.load_sovits(ss.MODEL_PATH, ss.CONFIG_PATH, dev)
audio = rd.render_cv(net_g, dev, cv, f0)        # 44.1 kHz float wav for this chunk
```

---

## 5. ContentVec → audio (the SVC backend)

`render_derisk.render_cv(net_g, dev, cv, f0)` is the verified glue:
1. `uv = (f0 < 30)` → unvoiced mask.
2. resample `cv` 50 fps → **SVC frame rate** (`SOVITS_FPS = 44100/512 ≈ 86.13` for so-vits 4.x), nearest.
3. resample `f0`, `uv` likewise; clamp f0 to [0, 1100] Hz.
4. `net_g.infer(c, f0, g=speaker_id, uv)` → waveform @ `SOVITS_SR = 44100`.

**Backends (all driven by the SAME cv; the backend = the voice):**
| backend | dim | feature | model | render path |
|---|---|---|---|---|
| so-vits-svc **4.1** | 768 | vec768l12 | 东雪莲 (`synth_sovits.MODEL_PATH`) | `render_derisk.render_cv` (validated default) |
| **RVC v2** | 768 | ContentVec | `lengv2.1.pth` | recipe below (zero-retrain, EAR-validated S59) |
| so-vits-svc **4.0** | 256 | vec256l9 | MinamiyaAkiko | `scripts/render_akiko.py` (needs the 256 cv) |

768 backends take `cv_final` directly. The 256 backend needs the 256 model (`cv256_final.pt`) — full guide in
**`runs/cv256/DEPLOY.md`**. ⚠ the specific akiko 4.0 used for S60 validation is low-quality (ja-biased, poor
cross-lingual); any decent vec256l9 4.0 voicebank decodes our 256 cv better (proven: GT features artifact on it too).

**RVC v2 recipe** (768 cv → RVC, no retrain; validated on `lengv2.1.pth`, S59):
```
RVC code: D:/MyDev/TESTING/Utai/RVC/RVC20240604Nvidia   (from infer_pack.models import SynthesizerTrnMs768NSFsid)
net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False);  net_g.load_state_dict(cpt["weight"], strict=False)
  # 103 "missing" keys = enc_q.* (train-only posterior) — expected, fine.
infer: feat = cv [T,768] @50fps ; f0(Hz,50fps) → upsample 2× to 100fps → f0_to_coarse → coarse pitch
       net_g.infer(feat_2x, p_len, coarse, nsff0=f0_2x, sid)  → audio @ sr=48000
```
The faiss `.index` is optional (not needed; index_rate=0 works). cv_final's 768 feeds straight in.

---

## 6. Conventions & gotchas

- **technique = all-zeros at deploy.** The `technique_proj` is zero-initialized (a no-op) AND all-zero is the
  dominant training condition (m4singer/ja-labs/ace are technique-free). Feeding zeros = in-distribution.
  (The 7 dims were a GTSinger expression channel, dims 0–3 force-zeroed in training; not used for deploy.)
- **cv is ~speaker-invariant.** `speaker_id` is a minor conditioning input; the *voice* is the SVC backend.
  Use a clean trained speaker (default 49 = kiritan). The id↔singer registry lives in the data-prep (each
  split record carries `speaker_id`+`dataset`); you rarely need it.
- **`lang_id` matters** — set it to the song's language (§3.2).
- **Frame = 20 ms (50 fps).** `T = sum(phone_dur)`. cv, f0, uv are all on this grid before the SVC resample.
- **Don't double-normalize.** `infer_cv` returns raw cv (already de-normalized). The norm stats are baked into
  the checkpoint as buffers (`cv_mean`/`cv_std`, loaded from `cv_norm_path`).
- **`note_pitch`/`note` clamp**: model clamps MIDI to [0,127]; SP/rest = 0.
- **Run under `.eval()` + `torch.no_grad()`**, `.float()` (fp32 inference).
- **render_ust.py's `--tau`/autopitch path uses the RETIRED learned f0 model** — for deployment use the
  **noteonly** f0 (or DAW parametric f0), NOT autopitch. See §7.

---

## 7. The f0 stream (separate; the DAW's job)

cv carries no pitch. You supply f0 (Hz, @50fps, aligned to cv; `uv` derived as `f0<30`).

- **The learned f0 model is RETIRED** (`runs/f0single`, not shipped). It undershot big interval jumps
  (e.g. a +12-semitone leap landed ~3.6 semitones flat); the error lives in the deterministic cents base and
  is τ-invariant, so it could not be tuned out. Decision (S59): f0 goes parametric in the DAW.
- **Deploy options:**
  - **noteonly** (use now): `f0 = 440·2^((note−69)/12)` at voiced frames, 0 at rests. Exact, mechanical,
    in-tune — the user-approved base. This is what the example in §4 uses.
  - **PARAMETRIC (the DAW target)**: noteonly base **+ deterministic portamento** (cosine slide *to* the target
    note over a few frames at transitions) **+ long-note-tail vibrato** (an LFO on the held tail). This is the
    SynthV-style hand-tuning that beats learned auto-pitch, and it's controllable. (A portamento prototype was
    validated in S58 → `processed/s58_porta`: slide the note target old→new with a cosine ramp over ~7 frames
    in log-pitch space at interval transitions.)
- Other DAW knobs that belong on the f0/expression side (not in this model): vibrato rate/depth, portamento
  width, formant/resonance shifts, etc.

---

## 8. Model architecture (reference)

```
phonemes ─embed(256)─┐
note_pitch ─embed(256)┤
speaker_id ─embed(256)├─concat─▶ Linear→512 ─▶ Conformer encoder (512, ×5) ─▶
lang_id ─embed(64)───┘                                                         │
technique ─Linear(7→256, zero-init, +to phone_emb)                            │
                                                                              ▼
                          soft Length Regulator (phone→frame expand, dur+pos features)
                                                                              ▼
                                       1-layer Conformer decoder (denoise jitter)
                                                                              ▼
                                    DetHead ──▶ E[cv | score]  (conditional mean)
```
- **~41 M params** (768) / ~40.96 M (256). Encoder dim 512, 5 layers; decoder 1 layer.
- **Deterministic head (`DetHead`)** = predicts the conditional-mean cv directly (no flow/diffusion/sampling).
  This is the "clean, elegant" core; FM/F0-head/CFG/stages were all stripped (S58).
- Output normalized internally; `infer_cv` de-normalizes.

---

## 9. Re-training / re-targeting (for a new backend feature)

The architecture is backend-agnostic — only the **target feature** changes. Recipe (proven S59–S60 for 256):
1. **Re-extract** the new target over the SAME audio (no re-align, no data change): a per-utt `.npy` feature
   set at 50 fps (`scripts/extract_contentvec*.py`). Confirm the exact checkpoint/layer/dim the backend expects.
2. **Build npz** = `scripts/build_npz256.py`-style: **reuse the existing npz fields (phonemes/dur/notes/f0)
   verbatim, swap ONLY the cv** to the frame-aligned new target. ⚠ **Do NOT rebuild npz from
   `manifest_final.jsonl`** — the manifest is STALE (predates the S46 dict-patch + S57 de-cram; those edits
   were never written back). Gate every clip: `cv.T == f0.T == Σphone_dur`.
3. **Retrain**: `scripts/train_cv.py --config <model_cvNNN.yaml>` with `cv_dim` set + the matching norm
   (`compute_cv_norm*.py`) + `npz_root` pointed at the new npz. ~60 k steps, ~2–3 h on a 3080 Ti.
4. **Validate by EAR** through the target backend before shipping (a det_floor number is NOT a ship gate).
- Objective = `det_loss` (normalized cv MSE = conditional mean) + small `pos_aux`. Quality signal =
  per-lang `det_floor` (cos to GT), but **EAR is ground truth**.

---

## 10. Quality / scope (honest)

- **7 languages EAR-accepted**: zh, ja, en, de, fr, es, it. (ko, ru dropped — bad alignment; embedding slots
  7/8 exist but unused.)
- `det_floor` plateaus ~**0.795** (768). Residual artifacts = minor 虚/糊 (breathy/blurred) consonants —
  an inherent property of a deterministic conditional-mean head, not under-training. `es` is a data-ceiling.
- The model sings a full real song end-to-end (銀の龍, `scripts/render_ust.py`). Judge per-language **by ear**
  (in-tune + intelligible), not by cv-space metrics — every cv-space metric in this project decoupled from EAR.
- The DAW frontend in `render_ust.py` is a FIRST-PASS heuristic (durations/grouping/chunking are not
  DAW-tuned). A real DAW supplies precise per-note timing and should expose the §3.3 / §7 knobs.
```
