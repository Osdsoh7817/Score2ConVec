"""Frontend reference: UST (SynthV / UTAU export) -> score arrays -> ScoreToCV -> SVC backend.
JP lyrics (kana OR romaji morae) -> IPA (matched to our ja inventory). Hand-built G2P (no pyopenjtalk).
  python scripts/render_ust.py --ust your_song.ust --dump               # 1) verify G2P / lyric mapping (no models needed)
  python scripts/render_ust.py --ust your_song.ust --out processed/out  # 2) render (needs cv_final.pt + a so-vits backend)
Set SOVITS_ROOT / SOVITS_MODEL / SOVITS_CONFIG before rendering (see README > Inference).
Deployment f0 = noteonly (exact note pitch); the learned f0 model is retired. See docs/DEPLOY_768_sovits41_rvc.md §7.
"""
import sys, re, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
from src.preprocessing.phoneme_vocab import PHONE_TO_ID
VOWEL_SET = {"a", "i", "ɯ", "e", "o"}

V = {"a": "a", "i": "i", "u": "ɯ", "e": "e", "o": "o"}
# romaji syllable -> IPA phones (matched to the ja-data inventory)
R2IPA = {
    "a": ["a"], "i": ["i"], "u": ["ɯ"], "e": ["e"], "o": ["o"],
    "ka": ["k","a"], "ki": ["k","i"], "ku": ["k","ɯ"], "ke": ["k","e"], "ko": ["k","o"],
    "ga": ["ɡ","a"], "gi": ["ɡ","i"], "gu": ["ɡ","ɯ"], "ge": ["ɡ","e"], "go": ["ɡ","o"],
    "sa": ["s","a"], "shi": ["ɕ","i"], "su": ["s","ɯ"], "se": ["s","e"], "so": ["s","o"], "si": ["ɕ","i"],
    "za": ["z","a"], "ji": ["dʑ","i"], "zu": ["z","ɯ"], "ze": ["z","e"], "zo": ["z","o"], "zi": ["dʑ","i"],
    "ta": ["t","a"], "chi": ["tɕ","i"], "tsu": ["ts","ɯ"], "te": ["t","e"], "to": ["t","o"], "ti": ["tɕ","i"], "tu": ["ts","ɯ"],
    "da": ["d","a"], "di": ["d","i"], "du": ["d","ɯ"], "de": ["d","e"], "do": ["d","o"],
    "na": ["n","a"], "ni": ["ɲ","i"], "nu": ["n","ɯ"], "ne": ["n","e"], "no": ["n","o"],
    "ha": ["h","a"], "hi": ["ç","i"], "fu": ["ɸ","ɯ"], "he": ["h","e"], "ho": ["h","o"], "hu": ["ɸ","ɯ"],
    "ba": ["b","a"], "bi": ["b","i"], "bu": ["b","ɯ"], "be": ["b","e"], "bo": ["b","o"],
    "pa": ["p","a"], "pi": ["p","i"], "pu": ["p","ɯ"], "pe": ["p","e"], "po": ["p","o"],
    "ma": ["m","a"], "mi": ["m","i"], "mu": ["m","ɯ"], "me": ["m","e"], "mo": ["m","o"],
    "ya": ["j","a"], "yu": ["j","ɯ"], "yo": ["j","o"],
    "ra": ["ɾ","a"], "ri": ["ɾ","i"], "ru": ["ɾ","ɯ"], "re": ["ɾ","e"], "ro": ["ɾ","o"],
    "wa": ["w","a"], "wo": ["o"], "n": ["ɴ"], "nn": ["ɴ"],
    "kya": ["c","a"], "kyu": ["c","ɯ"], "kyo": ["c","o"], "gya": ["ɟ","a"], "gyu": ["ɟ","ɯ"], "gyo": ["ɟ","o"],
    "sha": ["ɕ","a"], "shu": ["ɕ","ɯ"], "sho": ["ɕ","o"], "cha": ["tɕ","a"], "chu": ["tɕ","ɯ"], "cho": ["tɕ","o"],
    "ja": ["dʑ","a"], "ju": ["dʑ","ɯ"], "jo": ["dʑ","o"], "nya": ["ɲ","a"], "nyu": ["ɲ","ɯ"], "nyo": ["ɲ","o"],
    "hya": ["ç","a"], "hyu": ["ç","ɯ"], "hyo": ["ç","o"], "bya": ["bʲ","a"], "byu": ["bʲ","ɯ"], "byo": ["bʲ","o"],
    "pya": ["pʲ","a"], "pyu": ["pʲ","ɯ"], "pyo": ["pʲ","o"], "mya": ["mʲ","a"], "myu": ["mʲ","ɯ"], "myo": ["mʲ","o"],
    "rya": ["ɾʲ","a"], "ryu": ["ɾʲ","ɯ"], "ryo": ["ɾʲ","o"],
}
KANA = {  # hiragana -> romaji
    "あ":"a","い":"i","う":"u","え":"e","お":"o","か":"ka","き":"ki","く":"ku","け":"ke","こ":"ko",
    "が":"ga","ぎ":"gi","ぐ":"gu","げ":"ge","ご":"go","さ":"sa","し":"shi","す":"su","せ":"se","そ":"so",
    "ざ":"za","じ":"ji","ず":"zu","ぜ":"ze","ぞ":"zo","た":"ta","ち":"chi","つ":"tsu","て":"te","と":"to",
    "だ":"da","ぢ":"ji","づ":"zu","で":"de","ど":"do","な":"na","に":"ni","ぬ":"nu","ね":"ne","の":"no",
    "は":"ha","ひ":"hi","ふ":"fu","へ":"he","ほ":"ho","ば":"ba","び":"bi","ぶ":"bu","べ":"be","ぼ":"bo",
    "ぱ":"pa","ぴ":"pi","ぷ":"pu","ぺ":"pe","ぽ":"po","ま":"ma","み":"mi","む":"mu","め":"me","も":"mo",
    "や":"ya","ゆ":"yu","よ":"yo","ら":"ra","り":"ri","る":"ru","れ":"re","ろ":"ro","わ":"wa","を":"wo",
    "ん":"n","ぎゃ":"gya","きゃ":"kya","きゅ":"kyu","きょ":"kyo","しゃ":"sha","しゅ":"shu","しょ":"sho",
    "ちゃ":"cha","ちゅ":"chu","ちょ":"cho","にゃ":"nya","にゅ":"nyu","にょ":"nyo","ひゃ":"hya","ひょ":"hyo",
    "みゃ":"mya","りゃ":"rya","りゅ":"ryu","りょ":"ryo","じゃ":"ja","じゅ":"ju","じょ":"jo",
    "ぁ":"a","ぃ":"i","ぅ":"u","ぇ":"e","ぉ":"o",
}

def lyric_to_phones(lyr):
    """Return (list_of_ipa_phones, is_rest, is_sustain). C+V mora -> e.g. ['k','a']."""
    s = lyr.strip()
    if s in ("R", "r", "", "rest", "sil", "pau"): return [], True, False
    if s in ("-", "ー", "+"): return [], False, True       # sustain previous vowel
    if s in ("っ", "cl", "q", "っ"): return ["ʔ"], False, False
    # kana (possibly with a trailing small ゃゅょ already in KANA combos)
    if s in KANA: s = KANA[s]
    elif len(s) >= 2 and s[:2] in KANA: s = KANA[s[:2]]
    elif s[0] in KANA: s = KANA[s[0]]
    s = s.lower()
    if s in R2IPA: return list(R2IPA[s]), False, False
    # geminate: doubled leading consonant (tta/kke/ssa/ppa) = っ(ʔ) + mora
    if len(s) >= 3 and s[0] == s[1] and s[1:] in R2IPA:
        return ["ʔ"] + list(R2IPA[s[1:]]), False, False
    if s.startswith("tch") and ("ch" + s[3:]) in R2IPA:    # tchi -> っ ち
        return ["ʔ"] + list(R2IPA["ch" + s[3:]]), False, False
    return None, False, False    # signal unknown

def parse_ust(path):
    notes = []
    cur = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if re.match(r"^\[#", line):
            if cur and "Lyric" in cur: notes.append(cur)
            cur = {}
        elif "=" in line:
            k, _, v = line.partition("="); cur[k.strip()] = v.strip()
    if cur and "Lyric" in cur: notes.append(cur)
    return notes

TICK_PER_QUARTER = 480
def ust_to_score(path, fps=50):
    notes = parse_ust(path)
    tempo = 82.0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.startswith("Tempo="):
            try: tempo = float(raw.split("=")[1]); break
            except: pass
    frames_per_tick = (60.0 / tempo / TICK_PER_QUARTER) * fps
    phones, pdur, npitch, unknown = [], [], [], []
    raw_notes = []   # (notenum, frames, phones_for_this_note)
    for nt in notes:
        lyr = nt.get("Lyric", "R"); ln = int(nt.get("Length", "0")); nn = int(nt.get("NoteNum", "60"))
        fr = max(1, round(ln * frames_per_tick))
        ph, is_rest, is_sus = lyric_to_phones(lyr)
        raw_notes.append((lyr, nn, fr, ph, is_rest, is_sus))
        if ph is None: unknown.append(lyr)
    return raw_notes, tempo, frames_per_tick, unknown

def split_dur(fr, n):
    if n <= 1: return [max(1, fr)]
    c = min(4, max(1, fr // (n + 1)))             # each leading consonant ~ up to 4 frames, vowel gets the rest
    return [c] * (n - 1) + [max(1, fr - c * (n - 1))]

def build_arrays(raw_notes, cap_lead=25, cap_mid=70):
    phon, pdur, npitch = [], [], []
    prev_vowel = None
    M = len(raw_notes)
    for k, (lyr, nn, fr, ph, is_rest, is_sus) in enumerate(raw_notes):
        if is_rest:
            f = min(fr, cap_lead) if (k == 0 or k == M - 1) else min(fr, cap_mid)
            phon.append("SP"); pdur.append(max(1, f)); npitch.append(0); prev_vowel = None
        elif is_sus:
            v = prev_vowel or "a"
            phon.append(v); pdur.append(max(1, fr)); npitch.append(nn)
        else:
            for p, d in zip(ph, split_dur(fr, len(ph))):
                phon.append(p); pdur.append(d); npitch.append(nn)
            if ph and ph[-1] in VOWEL_SET: prev_vowel = ph[-1]
    phon_ids = [PHONE_TO_ID.get(p, PHONE_TO_ID["SP"]) for p in phon]
    n2p, nidx, prev = [], -1, None                # group consecutive same-pitch phones into one note
    for p in npitch:
        if p != prev: nidx += 1; prev = p
        n2p.append(nidx)
    from collections import defaultdict
    gf = defaultdict(int)
    for i, ni in enumerate(n2p): gf[ni] += pdur[i]
    ndur = [gf[ni] for ni in n2p]
    return (np.array(phon_ids), np.array(pdur), np.array(npitch), np.array(ndur), np.array(n2p), phon)

def render_song(ust, cv_spk, f0_spk, out, tau):
    import torch, yaml, soundfile
    from src.model.score2cv import ScoreToCV
    import render_derisk as rd, synth_sovits as ss
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        _ = torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev); torch.cuda.synchronize()
    cvm = ScoreToCV(yaml.safe_load(open(ROOT / "configs/model_cv_final.yaml", encoding="utf-8"))).to(dev).float().eval()
    cvm.load_state_dict(torch.load(ROOT / "runs/cvfinal/checkpoints/cv_final.pt", map_location=dev, weights_only=False)["model"])
    # The learned f0 model is RETIRED and not shipped. If its checkpoint is present we render an extra
    # "autopitch" take; otherwise we render "noteonly" f0 (exact note pitch) = the recommended deployment f0.
    f0m, f0_ckpt, f0_cfg = None, ROOT / "runs/f0single/checkpoints/step_10000.pt", ROOT / "runs/f0single/config.yaml"
    if f0_ckpt.exists() and f0_cfg.exists():
        from src.model.score2f0 import ScoreToF0
        f0m = ScoreToF0(yaml.safe_load(open(f0_cfg, encoding="utf-8"))).to(dev).float().eval()
        f0m.load_state_dict(torch.load(f0_ckpt, map_location=dev, weights_only=False)["model"])
    else:
        print("(no f0 model -> noteonly f0 only; this is the recommended deployment path)", flush=True)
    net_g, hps = ss.load_sovits(ss.MODEL_PATH, ss.CONFIG_PATH, dev)
    raw, *_ = ust_to_score(ust)
    pid, pdur, npitch, ndur, n2p, phon = build_arrays(raw)
    N = len(pid)
    # chunk at SP boundaries, ~<=400 frames each
    chunks, start, cf = [], 0, 0
    for i in range(N):
        cf += int(pdur[i])
        if cf > 400 and phon[i] == "SP":
            chunks.append((start, i + 1)); start = i + 1; cf = 0
    if start < N: chunks.append((start, N))
    print(f"score: {N} phones, {int(pdur.sum())} frames ({pdur.sum()/50:.1f}s), {len(chunks)} chunks", flush=True)

    def mk(s, e, spk):
        M = e - s
        z = lambda a: torch.tensor(a[s:e], dtype=torch.long, device=dev)[None]
        return dict(phonemes=z(pid), note_pitch=z(npitch), phone_dur=z(pdur), note_dur=z(ndur),
                    note_to_phone=torch.tensor(n2p[s:e] - n2p[s], dtype=torch.long, device=dev)[None],
                    speaker_id=torch.tensor([spk], device=dev), lang_id=torch.tensor([2], device=dev),
                    phone_mask=torch.ones(1, M, dtype=torch.bool, device=dev),
                    technique=torch.zeros(1, M, 7, device=dev))
    a_auto, a_note = [], []
    for (s, e) in chunks:
        with torch.no_grad():
            oc = cvm(**mk(s, e, cv_spk))
            T = int(oc["frame_mask"][0].sum())
            cv = cvm.infer_cv(oc["frame_hidden"])[0, :T].float().cpu().numpy()
        # noteonly f0 (exact note pitch per frame) — no model needed
        note = np.repeat(npitch[s:e], pdur[s:e]).astype(np.float32)[:T]
        note_hz = np.where(note > 0, 440.0 * 2.0 ** ((note - 69) / 12), 0.0).astype(np.float32)
        cv = cv[:len(note_hz)]
        a_note.append(rd.render_cv(net_g, dev, cv, note_hz))
        if f0m is not None:
            with torch.no_grad():
                fhf, fmf, midif = f0m.encode(**mk(s, e, f0_spk))
                f0a = f0m.infer_f0(fhf.float(), fmf, midif, tau=tau)[0][0, :T].float().cpu().numpy()
            a_auto.append(rd.render_cv(net_g, dev, cv, f0a[:len(note_hz)]))
        print(f"  chunk [{s}:{e}] T={T} done", flush=True)
    outdir = ROOT / out; outdir.mkdir(parents=True, exist_ok=True)
    variants = [("noteonly", a_note)] + ([("autopitch", a_auto)] if f0m is not None else [])
    for tag, parts in variants:
        w = np.concatenate(parts)
        w = (0.92 / (np.max(np.abs(w)) + 1e-9) * w).astype(np.float32)
        soundfile.write(str(outdir / f"render_{tag}.wav"), w, ss.SOVITS_SR)
    print(f"DONE -> {outdir}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ust", required=True)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv-spk", type=int, default=49)   # kiritan (clean reseg JA)
    ap.add_argument("--f0-spk", type=int, default=29)   # ace_acesinger_9 (the f0 model's singer)
    ap.add_argument("--tau", type=float, default=1.0)
    A = ap.parse_args()
    raw, tempo, fpt, unknown = ust_to_score(A.ust)
    print(f"UST: {len(raw)} notes | tempo={tempo} | frames/tick={fpt:.4f}")
    print("✓ all lyrics mapped" if not unknown else f"⚠ UNKNOWN ({len(unknown)}): {sorted(set(unknown))}")
    if A.dump:
        for lyr, nn, fr, ph, isr, iss in raw[:50]:
            tag = "REST" if isr else ("SUSTAIN" if iss else " ".join(ph) if ph else "??")
            print(f"  '{lyr:>4s}' note={nn:3d} fr={fr:3d} -> {tag}")
    if A.out:
        render_song(A.ust, A.cv_spk, A.f0_spk, A.out, A.tau)
