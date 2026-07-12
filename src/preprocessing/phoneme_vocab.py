"""Unified IPA phoneme vocabulary for Score2ContentVec v3.

Maps 9 languages' native phoneme formats to a single IPA token set.
Tokens 0-4: special.  Tokens 5+: IPA phonemes.

Source formats handled:
  - GTSinger (9 langs): strip language suffix + normalize tie bars
  - Opencpop-ext (ACE, M4Singer, PopCS): Chinese initials/finals → IPA
  - ARPABET (NUS-48E, GTSinger-EN): stress-stripped → IPA
  - JA-romaji (Itako, Kiritan, Oniku, Ofuton, Natsume, PJS): → IPA
  - CSD (EN/KR): syllable split → IPA
"""

import re
import unicodedata
from typing import Optional

# ── Special tokens ──────────────────────────────────────────────────

SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "EOS": 2, "SP": 3, "AP": 4}
NUM_SPECIAL = len(SPECIAL_TOKENS)
SP_ID = SPECIAL_TOKENS["SP"]
AP_ID = SPECIAL_TOKENS["AP"]
PAD_ID = SPECIAL_TOKENS["PAD"]

# ── Master IPA phoneme list ─────────────────────────────────────────
# Every IPA token that can appear in training data.
# Order is stable (determines ID assignment).  No duplicates.

IPA_PHONES = [
    # ─── Simple vowels ───
    "a", "e", "i", "o", "u",
    "ɯ",    # JA unrounded high back
    "y",    # FR/DE/ZH-ü
    "ɛ",    # DE FR IT KO RU
    "ɔ",    # DE ES FR IT
    "ɪ",    # DE EN RU
    "ʊ",    # DE EN RU
    "ɐ",    # DE KO RU
    "ə",    # DE FR RU EN
    "ɤ",    # ZH (Chinese standalone 'e')
    "æ",    # EN RU
    "ʌ",    # EN KO
    "ɑ",    # EN FR RU
    "ø",    # DE FR
    "œ",    # DE FR
    "ʉ",    # RU
    "ɵ",    # RU
    "ɨ",    # JA KO RU
    "ʏ",    # DE

    # ─── Long vowels ───
    "aː", "eː", "iː", "oː", "uː",
    "ɨː",  # JA KO
    "ɛː",  # KO
    "ʌː",  # KO

    # ─── French nasal vowels ───
    "ɛ̃", "ɔ̃", "ɑ̃",

    # ─── German long rounded vowels ───
    "øː", "yː",

    # ─── Devoiced vowels (JA) ───
    "i̥", "ɨ̥", "ɯ̥",

    # ─── Diphthongs (cross-language merged) ───
    "aɪ",   # ZH ai / EN AY / DE aj
    "eɪ",   # ZH ei / EN EY
    "aʊ",   # ZH ao / EN AW / DE aw
    "oʊ",   # ZH ou / EN OW
    "ɔɪ",   # EN OY
    "ɔʏ",   # DE

    # ─── Chinese compound finals ───
    "an", "ən", "ɑŋ", "əŋ", "ʊŋ",
    "ia", "iɛ", "iaʊ", "ioʊ", "iɛn", "in", "iɑŋ", "iŋ", "iʊŋ",
    "ua", "uo", "uaɪ", "ueɪ", "uan", "uən", "uɑŋ",
    "yɛn", "yɛ", "yn",
    "əɻ",

    # ─── Plosives ───
    "p", "b", "t", "d", "k", "ɡ",
    "pʰ", "tʰ", "kʰ",
    "ʔ",        # JA (cl / geminate)
    "c", "ɟ",   # palatal
    "cʰ",       # DE KO

    # ─── Affricates ───
    "ts", "tsʰ",
    "tɕ", "tɕʰ",
    "tʃ",
    "dʒ",
    "dʑ",
    "dz",
    "ʈʂ", "ʈʂʰ",  # ZH retroflex
    "pf",           # DE

    # ─── Fricatives ───
    "f", "v", "s", "z",
    "ʃ", "ʒ",
    "ɕ", "ʑ",
    "ç",
    "x",
    "ɣ",    # ES
    "h",
    "ʁ",    # DE FR
    "ʂ", "ʐ",  # RU ZH
    "ɸ",    # JA
    "β",    # ES
    "θ", "ð",
    "ɻ",    # ZH retroflex approximant
    "ʝ",    # ES
    "ɦ",    # KO

    # ─── Nasals ───
    "m", "n", "ŋ", "ɲ", "ɴ",

    # ─── Liquids ───
    "l", "ɾ", "r",
    "ɹ",    # EN
    "ɫ",    # RU
    "ɭ",    # KO
    "ʎ",    # ES FR IT RU

    # ─── Glides ───
    "w", "j",
    "ɥ",    # FR KO
    "ɰ",    # KO
    "ɰ̃",    # JA

    # ─── Palatalized ───
    "mʲ", "bʲ", "pʲ", "dʲ", "tʲ",
    "sʲ", "zʲ", "vʲ", "fʲ",
    "rʲ", "ɾʲ",
    "tsʲ",  # RU

    # ─── Geminates / long consonants ───
    "pː", "tː", "kː", "bː", "dː",
    "fː", "sː", "hː",
    "lː", "mː", "nː", "rː",
    "tʃː", "dʒː",
    "ɕː", "ʂː", "ʐː",
    "ɴː",

    # ─── Russian dental ───
    "n̪", "n̪ː",
    "t̪", "d̪",
    "s̪", "s̪ː", "z̪",
    "t̪s̪", "t̪s̪ː", "t̪ː",
    "ɲː",

    # ─── Korean fortis / unreleased / labialized ───
    "sʰ", "ɕʰ",
    "t͈", "k͈", "p͈",
    "s͈", "ɕ͈", "c͈",
    "tɕ͈",
    "p̚", "t̚", "k̚",
    "ɸʷ",
    "tɕʷ", "tʷ", "sʷ", "pʷ", "kʷ", "k͈ʷ",
    "t͈ʲ",

    # ─── German syllabic ───
    "m̩", "n̩",

    # ─── Spanish ───
    "ɟʝ",

    # ─── English rhotacized ───
    "ɝ",

    # ─── MFA additions ───
    "l̩",    # DE syllabic l
    "tɕː",  # RU
    "tʲː",  # RU

    # ─── Dict-eradication (S28) additions — APPENDED AT END so all existing IDs stay stable ───
    "ɹ̩",    # ZH apical DENTAL vowel  (zi/ci/si)        — A1, emitted by apply_dict_fixes
    "ɻ̩",    # ZH apical RETROFLEX vowel (zhi/chi/shi/ri) — A1
    "nʲ",   # RU soft н — palatalized alveolar (≠ palatal ɲ of es/it/fr) — C1
    "lʲ",   # RU soft л — palatalized alveolar (≠ palatal ʎ)             — C1
]

# ── Build lookups ───────────────────────────────────────────────────

PHONE_TO_ID: dict[str, int] = dict(SPECIAL_TOKENS)
for _i, _ph in enumerate(IPA_PHONES):
    assert _ph not in PHONE_TO_ID, f"Duplicate IPA phone: {_ph}"
    PHONE_TO_ID[_ph] = _i + NUM_SPECIAL

ID_TO_PHONE: dict[int, str] = {v: k for k, v in PHONE_TO_ID.items()}
VOCAB_SIZE = len(PHONE_TO_ID)

# ── GTSinger normalisation ──────────────────────────────────────────

_GTSINGER_NORMALIZE = {
    # Tie-bar variants (IT)
    "t͡ʃ": "tʃ", "t͡ʃː": "tʃː",
    "d͡ʒ": "dʒ", "d͡ʒː": "dʒː",
    "t͡s": "ts",
    # German diphthongs → cross-language merged
    "aj": "aɪ",
    "aw": "aʊ",
    # ASCII g → IPA ɡ (IT uses plain g)
    "g": "ɡ",
}

_GTSINGER_SPECIAL = {"<SP>": "SP", "<AP>": "AP"}

_LANG_SUFFIX = re.compile(r"_([a-z]{2})$")


def convert_gtsinger(phone: str) -> str:
    """GTSinger phone (e.g. 'ɕ_ja', '<SP>') → unified IPA string."""
    if phone in _GTSINGER_SPECIAL:
        return _GTSINGER_SPECIAL[phone]
    m = _LANG_SUFFIX.search(phone)
    if m:
        ipa = phone[: m.start()]
        lang = m.group(1)
    else:
        ipa = phone
        lang = None

    ipa = _GTSINGER_NORMALIZE.get(ipa, ipa)

    if lang == "zh":
        return _convert_gtsinger_zh(ipa)
    if lang == "en":
        return _convert_gtsinger_en(ipa)
    return ipa


def _convert_gtsinger_zh(phone: str) -> str:
    """GTSinger ZH uses opencpop-extension notation → IPA."""
    if phone in OPENCPOP_INITIALS:
        return OPENCPOP_INITIALS[phone]
    if phone in OPENCPOP_FINALS:
        return OPENCPOP_FINALS[phone]
    return phone


def _convert_gtsinger_en(phone: str) -> str:
    """GTSinger EN uses ARPABET → IPA."""
    return convert_arpabet(phone)


# ── Opencpop (ACE / M4Singer / PopCS) ──────────────────────────────

OPENCPOP_INITIALS = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f",
    "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x",
    "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ɻ",
    "z": "ts", "c": "tsʰ", "s": "s",
    "w": "w", "y": "j",  # onset glides (ACE-Opencpop notation)
}

OPENCPOP_FINALS = {
    "a": "a", "o": "o", "e": "ɤ", "i": "i", "u": "u", "v": "y",
    "ai": "aɪ", "ei": "eɪ", "ao": "aʊ", "ou": "oʊ",
    "an": "an", "en": "ən", "ang": "ɑŋ", "eng": "əŋ", "ong": "ʊŋ",
    "ia": "ia", "ie": "iɛ", "iao": "iaʊ", "iou": "ioʊ",
    "ian": "iɛn", "in": "in", "iang": "iɑŋ", "ing": "iŋ", "iong": "iʊŋ",
    "ua": "ua", "uo": "uo", "uai": "uaɪ", "uei": "ueɪ",
    "uan": "uan", "uen": "uən", "uang": "uɑŋ",
    "van": "yɛn", "ve": "yɛ", "vn": "yn",
    "er": "əɻ",
    # ACE-Opencpop simplified aliases
    "iu": "ioʊ", "un": "uən", "ui": "ueɪ",
}

_OPENCPOP_SPECIAL = {"SP": "SP", "AP": "AP", "<SP>": "SP", "<AP>": "AP"}


def convert_opencpop(phone: str) -> str:
    """Opencpop-extension phone → unified IPA string.

    Handles both initials (b/p/m/f/...) and finals (a/ai/uan/...).
    """
    if phone in _OPENCPOP_SPECIAL:
        return _OPENCPOP_SPECIAL[phone]
    if phone in OPENCPOP_INITIALS:
        return OPENCPOP_INITIALS[phone]
    if phone in OPENCPOP_FINALS:
        return OPENCPOP_FINALS[phone]
    return phone


# ── ARPABET (NUS-48E, GTSinger-EN) ─────────────────────────────────

ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AX": "ə",
    "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
    "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m",
    "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ",
    "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

_ARPABET_STRESS = re.compile(r"[012]$")
_ARPABET_SPECIAL = {"sil": "SP", "sp": "SP", "spn": "SP"}


def convert_arpabet(phone: str) -> str:
    """ARPABET phone (with optional stress) → unified IPA string.

    AH is special (A2 fix): stress encodes a phonemic split that the generic
    stress-strip would destroy — AH0 = schwa [ə], AH1/AH2 = STRUT [ʌ]
    (love/up/cup).  GTSinger-EN uses CMUdict AH0/AH1 (there is NO separate 'AX'
    symbol), so AH must be resolved BEFORE stress-stripping.
    """
    if phone.lower() in _ARPABET_SPECIAL:
        return _ARPABET_SPECIAL[phone.lower()]
    up = phone.upper()
    if up.startswith("AH"):
        return "ə" if up[2:] == "0" else "ʌ"   # AH0→schwa, AH1/AH2/bare→STRUT
    base = _ARPABET_STRESS.sub("", up)
    if base in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[base]
    return phone


# ── Japanese romaji (Itako, Kiritan, Oniku, Ofuton, Natsume, PJS) ──

JA_ROMAJI_TO_IPA = {
    # Vowels
    "a": "a", "i": "i", "u": "ɯ", "e": "e", "o": "o",
    # Devoiced vowels (uppercase in some datasets: Oniku, Ofuton, Natsume)
    "I": "i̥", "U": "ɯ̥",
    # Consonants
    "k": "k", "g": "ɡ", "s": "s", "z": "z",
    "t": "t", "d": "d", "n": "n", "h": "h",
    "b": "b", "p": "p", "m": "m",
    "r": "ɾ", "w": "w", "y": "j",
    "v": "v",  # loan words
    # Compound consonants
    "sh": "ɕ", "ch": "tɕ",
    "ts": "ts", "dz": "dz",
    "f": "ɸ",
    "j": "dʑ",
    "ky": "c", "gy": "ɟ",
    "ty": "c", "dy": "ɟ",  # palatalized t/d (same as ky/gy)
    "ny": "ɲ", "hy": "ç",
    "by": "bʲ", "py": "pʲ", "my": "mʲ", "ry": "ɾʲ",
    # Special phones
    "N": "ɴ",
    "cl": "ʔ",
    "q": "ʔ",
    # Silence / breath / filler
    "pau": "SP", "sil": "SP", "br": "AP",
    "xx": "SP",  # PJS filler/noise
}

# Longest-first matching for compound consonants
_JA_SORTED_KEYS = sorted(JA_ROMAJI_TO_IPA.keys(), key=len, reverse=True)


def convert_ja_romaji(phone: str) -> str:
    """Japanese lab romaji phone → unified IPA string."""
    if phone in JA_ROMAJI_TO_IPA:
        return JA_ROMAJI_TO_IPA[phone]
    return phone


# ── MFA (FR/DE/IT/KR/RU/ES — raw IPA, needs normalization) ────────

_MFA_NORMALIZE = {
    **_GTSINGER_NORMALIZE,
    "tʂː": "ʂː",
}


def convert_mfa(phone: str) -> str:
    """MFA IPA phone → unified IPA string."""
    if not phone or phone == "spn":
        return "SP"
    phone = unicodedata.normalize("NFC", phone)
    return _MFA_NORMALIZE.get(phone, phone)


# ── CSD (Children's Song Dataset — EN/KR) ──────────────────────────
# CSD labels are syllable-level with onset_nucleus notation.
# We split into onset + nucleus and convert each.

CSD_EN_ONSET = {
    "b": "b", "d": "d", "f": "f", "g": "ɡ", "h": "h",
    "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n",
    "p": "p", "r": "ɹ", "s": "s", "t": "t", "v": "v",
    "w": "w", "y": "j", "z": "z",
    "th": "θ", "dh": "ð", "sh": "ʃ", "zh": "ʒ",
    "ch": "tʃ", "ng": "ŋ",
}

CSD_EN_NUCLEUS = {
    "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
    "aw": "aʊ", "ay": "aɪ",
    "eh": "ɛ", "er": "ɝ", "ey": "eɪ",
    "ih": "ɪ", "ii": "i", "iy": "i",
    "ow": "oʊ", "oy": "ɔɪ",
    "uh": "ʊ", "uw": "u",
}

CSD_KR_ONSET = {
    "g": "k", "kk": "k͈", "k": "kʰ",
    "d": "t", "tt": "t͈", "t": "tʰ",
    "b": "p", "pp": "p͈", "p": "pʰ",
    "s": "s", "ss": "s͈",
    "j": "tɕ", "jj": "tɕ͈", "ch": "tɕʰ",
    "n": "n", "m": "m", "l": "ɾ",
    "r": "ɾ", "h": "h",
}

CSD_KR_NUCLEUS = {
    "a": "a", "e": "e", "eo": "ʌ", "eu": "ɨ",
    "i": "i", "o": "o", "u": "u",
    "ae": "ɛ", "oe": "ø", "wi": "y",
    "wa": "ua", "wae": ["w", "ɛ"], "wo": "uo",
    "we": ["w", "e"], "ya": "ia", "ye": "iɛ",
    "yo": ["j", "o"], "yu": ["j", "u"], "yeo": ["j", "ʌ"],
    "yae": "iɛ",
}


def convert_csd(syllable: str, lang: str = "en") -> list[str]:
    """CSD syllable (e.g. 'b_ii', 'ss_i') → list of IPA strings."""
    if "_" in syllable:
        onset, nucleus = syllable.split("_", 1)
    else:
        onset, nucleus = "", syllable

    result = []
    if lang == "en":
        if onset:
            result.append(CSD_EN_ONSET.get(onset, onset))
        nuc = CSD_EN_NUCLEUS.get(nucleus, nucleus)
    elif lang == "ko":
        if onset:
            result.append(CSD_KR_ONSET.get(onset, onset))
        nuc = CSD_KR_NUCLEUS.get(nucleus, nucleus)
    else:
        return result
    # nucleus value may be a single IPA string or a glide+vowel list (KR yo/yu/yeo/wae/we)
    result.extend(nuc if isinstance(nuc, list) else [nuc])
    return result


# ── M4Singer / PopCS TextGrid phones ───────────────────────────────
# These use opencpop-style phone notation split into individual phones
# in the TextGrid tiers.  Map through convert_opencpop.

def convert_m4singer(phone: str) -> str:
    """M4Singer TextGrid phone → unified IPA string."""
    return convert_opencpop(phone)


def convert_popcs(phone: str) -> str:
    """PopCS TextGrid phone → unified IPA string."""
    if phone == "" or phone.strip() == "":
        return "SP"
    return convert_opencpop(phone)


# ── Unified dispatcher ──────────────────────────────────────────────

_CONVERTERS = {
    "gtsinger": convert_gtsinger,
    "opencpop": convert_opencpop,
    "ace": convert_opencpop,
    "m4singer": convert_m4singer,
    "popcs": convert_popcs,
    "arpabet": convert_arpabet,
    "nus48e": convert_arpabet,
    "ja_romaji": convert_ja_romaji,
    "itako": convert_ja_romaji,
    "kiritan": convert_ja_romaji,
    "oniku": convert_ja_romaji,
    "ofuton": convert_ja_romaji,
    "natsume": convert_ja_romaji,
    "pjs": convert_ja_romaji,
    "mfa": convert_mfa,
}


def convert(phone: str, source: str) -> str | list[str]:
    """Convert a phone from any source format to unified IPA.

    Args:
        phone: phone string in source format
        source: one of the keys in _CONVERTERS, or 'csd_en'/'csd_ko'

    Returns:
        IPA string (or list for CSD which splits syllables)
    """
    if source == "csd_en":
        return convert_csd(phone, "en")
    if source == "csd_ko":
        return convert_csd(phone, "ko")
    converter = _CONVERTERS.get(source)
    if converter is None:
        raise ValueError(f"Unknown source format: {source}")
    return converter(phone)


def phone_to_id(ipa: str) -> int:
    """IPA string → token ID.  Returns PAD_ID for unknown phones."""
    return PHONE_TO_ID.get(ipa, PAD_ID)


def phones_to_ids(phones: list[str]) -> list[int]:
    """Convert list of IPA strings to token IDs."""
    return [phone_to_id(p) for p in phones]


def ids_to_phones(ids: list[int]) -> list[str]:
    """Convert token IDs back to IPA strings."""
    return [ID_TO_PHONE.get(i, "?") for i in ids]


# ── Validation ──────────────────────────────────────────────────────

def _validate():
    """Run at import time: check all converter outputs are in vocab."""
    missing = set()

    # Check opencpop
    for table in (OPENCPOP_INITIALS, OPENCPOP_FINALS):
        for src, ipa in table.items():
            if ipa not in PHONE_TO_ID:
                missing.add(("opencpop", src, ipa))

    # Check ARPABET
    for src, ipa in ARPABET_TO_IPA.items():
        if ipa not in PHONE_TO_ID:
            missing.add(("arpabet", src, ipa))

    # Check JA romaji
    for src, ipa in JA_ROMAJI_TO_IPA.items():
        if ipa not in PHONE_TO_ID and ipa not in ("SP", "AP"):
            missing.add(("ja_romaji", src, ipa))

    # Check CSD (dead converters — no training data — but guard against silent PAD)
    for table in (CSD_EN_ONSET, CSD_EN_NUCLEUS, CSD_KR_ONSET, CSD_KR_NUCLEUS):
        for src, ipa in table.items():
            for tok in (ipa if isinstance(ipa, list) else [ipa]):
                if tok not in PHONE_TO_ID:
                    missing.add(("csd", src, tok))

    # Dict-eradication (S28): the appended tokens MUST exist (apply_dict_fixes emits them)
    for tok in ("ɹ̩", "ɻ̩", "nʲ", "lʲ"):
        if tok not in PHONE_TO_ID:
            missing.add(("dict-add", tok, tok))

    if missing:
        raise RuntimeError(
            f"Phoneme vocab validation failed — {len(missing)} converter outputs "
            f"not in IPA_PHONES:\n"
            + "\n".join(f"  {src_fmt} '{src}' → '{ipa}'" for src_fmt, src, ipa in sorted(missing))
        )


_validate()
