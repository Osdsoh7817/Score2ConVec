"""Context-aware dictionary fixes (S28 eradication) at the SEQUENCE level.

The per-phone converters in `phoneme_vocab.py` cannot see context or language;
these fixes need the previous/next phone and/or the language, so they live here
as a single source of truth used by BOTH the data pipeline
(`convert_alignments.py`), the in-place npz patcher, AND the inference
score→phone path — so training and inference can never diverge.

Two entry points:
  * `apply_dict_fixes(phones, lang)` — LENGTH-PRESERVING remaps:
        A1 (zh apical-i split), C1 (ru soft cons), C2 (non-ja palatal-stop
        de-narrow), C3 (dead-token cleanup), D (ja ひ allophone).
  * `split_ko_labialized(phones, durs)` — the ONLY LENGTH-CHANGING fix:
        C4 (ko Cʷ → C + w).  Returned `dup_map` lets the caller replicate any
        parallel per-phone array (note_pitch / note_dur / note_to_phone /
        technique) for the inserted [w].

NOT handled here:
  * A2 (en AH stress) — fixed in `phoneme_vocab.convert_arpabet` for the
    pipeline/inference; the stress digit is LOST in already-processed data, so
    the npz side re-derives it from the raw GTSinger-EN source separately.
  * A3 (van→yɛn) — a pure rename in `phoneme_vocab` (token ID unchanged).

Decision log (grounded in real npz per-(lang,token) counts):
  A1  zh 'i' #1 content token; onset∈{ts,tsʰ,s}→ɹ̩ (apical dental),
      ∈{ʈʂ,ʈʂʰ,ʂ,ɻ}→ɻ̩ (apical retroflex); else true close-front [i].
  C1  ru ɲ=2203, ʎ=1507 (HIGH) — Russian soft н/л are palatalized ALVEOLAR
      (nʲ/lʲ), not the palatal ɲ/ʎ of es/it/fr → must not share the token.
  C2  non-ja [c]/[ɟ]/[cʰ] = MFA allophonic velar-fronting before front vowels
      (es que=[ce]); de-narrow to k/ɡ/kʰ (the fronting is recoverable from the
      following front vowel).  KEEP ja [c]/[ɟ] (phonemic きゃ/ぎゃ) + ALL [ç]
      (real fricative; de ich-laut=2973).
  C3  ONLY tokens that are GLOBALLY near-dead (cannot be learned) merge into
      their nearest base — NOT tokens that are rare in one language but shared
      and well-populated globally (fr ŋ=8 / fr·de tʃ stay: ŋ,tʃ have thousands
      cross-lingually = the whole point of unified IPA).
        global-dead → base (all langs): l̩→l, bː→b, dː→d, t͈ʲ→t͈, c͈→k͈
          (it bː/dː are REAL Italian geminates but only 17/9 occurrences →
           unlearnable as own token; merge + documented. cf. ja gemination is
           frequent+phonemic → PRESERVED via re-align, not here.)
        lang-specific noise (token stays alive via another lang): ko dʑ→tɕ
          (dʑ alive via ja=194), ko mː→m (mː alive via it=78).
  C4  ko labialized kʷ/tʷ/pʷ/sʷ/tɕʷ/ɸʷ/k͈ʷ → C + [w] (every other language
      writes the w-glide separately).  LENGTH-CHANGING → split_ko_labialized.
  D   clean DETERMINISTIC subset only: ja ひ — /h/ before /i/(/i̥/) = [ç].
      SKIPPED (fragile heuristic OR info we don't have): en dark-l/ER/flapping,
      zh o→ɔ/wo (needs reliable context), zh neutral-tone e→ə (no tone in the
      labels), ja が-row [ŋ] (speaker/dialect-dependent, not text-derivable).
"""
from __future__ import annotations

# ── A1: zh apical-i split (onset-conditioned) ───────────────────────
_A1_DENTAL = {"ts", "tsʰ", "s"}            # zi/ci/si           → ɹ̩
_A1_RETRO = {"ʈʂ", "ʈʂʰ", "ʂ", "ɻ"}       # zhi/chi/shi/ri      → ɻ̩

# ── C1: ru soft consonants (palatalized alveolar ≠ palatal) ─────────
_C1_RU = {"ɲ": "nʲ", "ʎ": "lʲ"}

# ── C2: non-ja palatal-stop de-narrow (allophonic velar-fronting) ───
_C2 = {"c": "k", "ɟ": "ɡ", "cʰ": "kʰ"}

# ── C3: dead-token cleanup ──────────────────────────────────────────
_C3_GLOBAL = {"l̩": "l", "bː": "b", "dː": "d", "t͈ʲ": "t͈", "c͈": "k͈"}
_C3_LANG = {"ko": {"dʑ": "tɕ", "mː": "m"}}

# ── C4: ko labialized consonants → C + w (LENGTH-CHANGING) ──────────
_C4_KO = {"kʷ": "k", "tʷ": "t", "pʷ": "p", "sʷ": "s",
          "tɕʷ": "tɕ", "ɸʷ": "ɸ", "k͈ʷ": "k͈"}

# ── D: ja ひ allophone (/h/ before /i/ → ç) ─────────────────────────
_D_JA_HI_FOLLOWERS = {"i", "i̥"}


def apply_dict_fixes(phones, lang: str) -> list:
    """Length-preserving sequence remaps (A1/C1/C2/C3/D). Returns new phone list.

    Context (prev/next phone) is read from the ORIGINAL `phones` — none of the
    onsets/followers these rules key on are themselves rewritten by another
    rule, so original indexing is safe and order-independent.
    """
    n = len(phones)
    out = []
    for i in range(n):
        ph = phones[i]

        # A1 — zh apical-i split (look at the original previous phone = onset)
        if lang == "zh" and ph == "i":
            prev = phones[i - 1] if i > 0 else None
            if prev in _A1_DENTAL:
                out.append("ɹ̩"); continue
            if prev in _A1_RETRO:
                out.append("ɻ̩"); continue
            out.append("i"); continue

        # C2 — de-narrow non-ja palatal stops (KEEP ja phonemic c/ɟ)
        if lang != "ja" and ph in _C2:
            out.append(_C2[ph]); continue

        # C1 — ru soft consonants
        if lang == "ru" and ph in _C1_RU:
            out.append(_C1_RU[ph]); continue

        # C3 — global dead-token cleanup
        if ph in _C3_GLOBAL:
            out.append(_C3_GLOBAL[ph]); continue

        # C3 — lang-specific noise (token alive via another lang)
        if lang in _C3_LANG and ph in _C3_LANG[lang]:
            out.append(_C3_LANG[lang][ph]); continue

        # D — ja ひ: /h/ before /i/(/i̥/) → ç
        if lang == "ja" and ph == "h" and i + 1 < n and phones[i + 1] in _D_JA_HI_FOLLOWERS:
            out.append("ç"); continue

        out.append(ph)
    return out


def split_ko_labialized(phones, durs, lang: str):
    """C4 — split ko Cʷ → C + w (the ONLY length-changing fix).

    Returns (new_phones, new_durs, dup_map):
      * new_phones / new_durs : expanded lists (one extra entry per Cʷ).
      * dup_map : len(new) ints; dup_map[k] = the ORIGINAL phone index that
        output position k derives from.  A parallel per-phone array A (note
        pitch/dur/note_to_phone/technique) is replicated as A[dup_map] — the
        inserted [w] inherits the source consonant's note/technique.

    Duration split: the [w] glide is a short trailing portion (≈⅓, ≥1 frame),
    the base consonant keeps the rest; total is conserved.  If the Cʷ is a
    single frame (cannot make two ≥1-frame phones) the [w] is dropped and the
    phone is just relabeled to its base (rare edge; frame budget preserved).
    """
    if lang != "ko":
        return list(phones), list(durs), list(range(len(phones)))

    new_ph, new_du, dup = [], [], []
    for i, ph in enumerate(phones):
        du = int(durs[i])
        if ph in _C4_KO:
            base = _C4_KO[ph]
            if du >= 2:
                w_du = max(1, du // 3)
                base_du = du - w_du
                new_ph.append(base); new_du.append(base_du); dup.append(i)
                new_ph.append("w"); new_du.append(w_du); dup.append(i)
            else:  # du == 1: can't split → relabel to base, drop the w
                new_ph.append(base); new_du.append(du); dup.append(i)
        else:
            new_ph.append(ph); new_du.append(du); dup.append(i)
    return new_ph, new_du, dup
