"""
dataset/phonetic_mapping.py — Phonetic label → Brahmi Unicode mapping
=====================================================================

Maps the romanized / IAST-style phonetic folder names used by the
Capstone_Brahmi_Inscriptions and BrahmiGAN datasets to the correct
Brahmi Unicode character(s) (U+11000 – U+1107F).

The mapping covers all 287 class folders from the Capstone
RecognizerDataset_150_210 dataset, plus the 23 base-character
folders from the BrahmiGAN dataset.

USAGE:
    from dataset.phonetic_mapping import phonetic_to_brahmi
    brahmi_text = phonetic_to_brahmi("kaa")   # → "𑀓𑀸"
"""

# ---------------------------------------------------------------------------
# Brahmi Unicode code-point constants
# ---------------------------------------------------------------------------

# Independent vowels
_V = {
    "a":   "\U00011005",   # 𑀅
    "aa":  "\U00011006",   # 𑀆
    "i":   "\U00011007",   # 𑀇
    "ii":  "\U00011008",   # 𑀈
    "u":   "\U00011009",   # 𑀉
    "uu":  "\U0001100A",   # 𑀊
    "r":   "\U0001100B",   # 𑀋  (vocalic r)
    "rr":  "\U0001100C",   # 𑀌  (vocalic rr)
    "l":   "\U0001100D",   # 𑀍  (vocalic l)
    "ll":  "\U0001100E",   # 𑀎  (vocalic ll)
    "e":   "\U0001100F",   # 𑀏
    "ai":  "\U00011010",   # 𑀐
    "o":   "\U00011011",   # 𑀑
    "au":  "\U00011012",   # 𑀒
}

# Consonants (inherent "a" vowel)
_C = {
    "ka":   "\U00011013",  # 𑀓
    "kha":  "\U00011014",  # 𑀔
    "ga":   "\U00011015",  # 𑀕
    "gha":  "\U00011016",  # 𑀖
    "nga":  "\U00011017",  # 𑀗
    "ca":   "\U00011018",  # 𑀘
    "cha":  "\U00011019",  # 𑀙
    "ja":   "\U0001101A",  # 𑀚
    "jha":  "\U0001101B",  # 𑀛
    "nya":  "\U0001101C",  # 𑀜  (ña)
    "tta":  "\U0001101D",  # 𑀝  (retrofl. ṭa)
    "ttha": "\U0001101E",  # 𑀞  (retrofl. ṭha)
    "dda":  "\U0001101F",  # 𑀟  (retrofl. ḍa)
    "ddha": "\U00011020",  # 𑀠  (retrofl. ḍha)
    "nna":  "\U00011021",  # 𑀡  (retrofl. ṇa)
    "ta":   "\U00011022",  # 𑀢
    "tha":  "\U00011023",  # 𑀣
    "da":   "\U00011024",  # 𑀤
    "dha":  "\U00011025",  # 𑀥
    "na":   "\U00011026",  # 𑀦
    "pa":   "\U00011027",  # 𑀧
    "pha":  "\U00011028",  # 𑀨
    "ba":   "\U00011029",  # 𑀩
    "bha":  "\U0001102A",  # 𑀪
    "ma":   "\U0001102B",  # 𑀫
    "ya":   "\U0001102C",  # 𑀬
    "ra":   "\U0001102D",  # 𑀭
    "la":   "\U0001102E",  # 𑀮
    "va":   "\U0001102F",  # 𑀯
    "sha":  "\U00011030",  # 𑀰  (śa)
    "ssa":  "\U00011031",  # 𑀱  (ṣa)
    "sa":   "\U00011032",  # 𑀲
    "ha":   "\U00011033",  # 𑀳
}

# Dependent vowel signs (combine with consonants)
_VS = {
    "aa":  "\U00011038",   # 𑀸  (ā sign)
    "i":   "\U00011039",   # 𑀹
    "ii":  "\U0001103A",   # 𑀺
    "u":   "\U0001103B",   # 𑀻
    "uu":  "\U0001103C",   # 𑀼
    "r":   "\U0001103D",   # 𑀽  (vocalic r sign)
    "rr":  "\U0001103E",   # 𑀾  (vocalic rr sign)
    "e":   "\U00011040",   # 𑁀
    "ai":  "\U00011041",   # 𑁁
    "o":   "\U00011042",   # 𑁂
    "au":  "\U00011043",   # 𑁃
}

# ---------------------------------------------------------------------------
# Vowel-suffix patterns (longest match first)
# When a folder name like "kaa" is encountered, we split off the consonant
# root "ka" and the vowel suffix "a" → which indicates the ā sign.
#
# Conventions used by the Capstone dataset:
#   folder "ka"   = consonant with inherent 'a'   → just the consonant
#   folder "kaa"  = consonant + ā vowel sign       → consonant + _VS["aa"]
#   folder "ki"   = consonant + i vowel sign        → consonant + _VS["i"]
#   folder "kii"  = consonant + ī vowel sign        → consonant + _VS["ii"]
#   folder "ku"   = consonant + u vowel sign        → consonant + _VS["u"]
#   folder "kuu"  = consonant + ū vowel sign        → consonant + _VS["uu"]
#   folder "ke"   = consonant + e vowel sign        → consonant + _VS["e"]
#   folder "ko"   = consonant + o vowel sign        → consonant + _VS["o"]
# ---------------------------------------------------------------------------

# Vowel suffixes, ordered longest-first for matching
_VOWEL_SUFFIXES = [
    ("uu", "uu"),
    ("ii", "ii"),
    ("aa", "aa"),
    ("ai", "ai"),
    ("au", "au"),
    ("u",  "u"),
    ("i",  "i"),
    ("e",  "e"),
    ("o",  "o"),
]


def _build_mapping() -> dict:
    """
    Build the complete mapping from Capstone phonetic folder names
    to Brahmi Unicode strings.
    """
    mapping = {}

    # --- Pure vowels ---
    mapping["a"]  = _V["a"]
    mapping["i"]  = _V["i"]
    mapping["e"]  = _V["e"]
    mapping["o"]  = _V["o"]
    mapping["u"]  = _V["u"]
    mapping["ee"] = _V["e"]       # alternate for e
    mapping["aaa"] = _V["aa"]     # long ā
    # Capstone uses extra "a"s for long vowels in some classes:
    # a(3), a(4), a(5) appear to be variant forms of 'a'
    mapping["a(3)"] = _V["a"]
    mapping["a(4)"] = _V["a"]
    mapping["a(5)"] = _V["a"]
    mapping["o(2)"] = _V["o"]

    # --- Consonants: base form (inherent 'a') ---
    for phonetic_name, brahmi_char in _C.items():
        mapping[phonetic_name] = brahmi_char

    # --- Consonant variants with (2), (3), (4) ---
    # These are allographic variants (different writing styles of the same
    # consonant). We map them to the same Brahmi character.
    variant_base = {
        "ba(2)": "ba", "bo(2)": "ba",  # ba variants
        "da(2)": "da",                  # da variant
        "daa(2)": "da",                # da variant (will be overwritten below)
        "ja(2)": "ja", "ja(3)": "ja", "ja(4)": "ja",
        "ka(2)": "ka",
        "kha(2)": "kha", "khaa(2)": "kha", "khe(2)": "kha",
        "khii(2)": "kha", "kho(2)": "kha", "khu(2)": "kha", "khuu(2)": "kha",
        "la(2)": "la", "la(3)": "la",
        "ma(2)": "ma",
        "na(2)": "na",
        "nno(2)": "nna",
        "nya(2)": "nya",
        "pha(2)": "pha",
        "ra(2)": "ra", "ra(3)": "ra",
        "sa(2)": "sa",
        "tha(2)": "tha", "the(2)": "tha",
        "vu(2)": "va", "vuu(2)": "va",
        "ya(2)": "ya", "yo(2)": "ya",
    }

    # Map base consonant variants (e.g. "da(2)" → same as "da")
    for variant, base in variant_base.items():
        if base in _C:
            mapping[variant] = _C[base]

    # --- Consonant + vowel sign combinations ---
    # For each consonant, generate all vowel-sign combinations
    for cons_name, cons_char in _C.items():
        for suffix, vs_key in _VOWEL_SUFFIXES:
            if vs_key in _VS:
                combo_name = cons_name[:-1] + suffix  # strip trailing 'a', add suffix
                # Only add if not already present (avoid overwriting base forms)
                if combo_name not in mapping and combo_name != cons_name:
                    mapping[combo_name] = cons_char + _VS[vs_key]

    # --- Special Capstone naming conventions ---
    # The Capstone dataset uses some non-standard romanization:

    # "daaa" / "daaaa" — extra 'a's for emphasis or long vowel variants
    # We treat these as consonant + long vowel sign
    for cons_name, cons_char in _C.items():
        base = cons_name[:-1]  # strip trailing 'a'
        # triple-a → ā sign (same as double)
        triple = base + "aaa"
        if triple not in mapping:
            mapping[triple] = cons_char + _VS["aa"]
        # quadruple-a → ā sign
        quad = base + "aaaa"
        if quad not in mapping:
            mapping[quad] = cons_char + _VS["aa"]

    # "dhue" → dhu + e? Treat as dha + u sign (closest match)
    mapping["dhue"] = _C["dha"] + _VS["u"]

    # "vhu" / "vhuu" — likely a dataset-specific notation for bha variants
    mapping["vhu"]  = _C["bha"] + _VS["u"]
    mapping["vhuu"] = _C["bha"] + _VS["uu"]

    # Capstone "thaai" — tha + ai vowel sign
    mapping["thaai"] = _C["tha"] + _VS["ai"]

    # Fix up variant + vowel combinations
    # e.g. "daa(2)" should be da + ā
    mapping["daa(2)"] = _C["da"] + _VS["aa"]
    mapping["bo(2)"]  = _C["ba"] + _VS["o"]
    mapping["nno(2)"] = _C["nna"] + _VS["o"]
    mapping["the(2)"] = _C["tha"] + _VS["e"]
    mapping["yo(2)"]  = _C["ya"] + _VS["o"]
    mapping["vu(2)"]  = _C["va"] + _VS["u"]
    mapping["vuu(2)"] = _C["va"] + _VS["uu"]

    # Capstone-specific vowel-suffix edge cases
    # "tai", "taii" — ta + ai sign, ta + long-i sign
    mapping["tai"]   = _C["ta"] + _VS["ai"]
    mapping["taii"]  = _C["ta"] + _VS["ii"]
    mapping["tae"]   = _C["ta"] + _VS["e"]
    mapping["tao"]   = _C["ta"] + _VS["o"]
    mapping["tau"]   = _C["ta"] + _VS["u"]
    mapping["tauu"]  = _C["ta"] + _VS["uu"]

    # "dai", "daii", etc
    mapping["dai"]   = _C["da"] + _VS["ai"]
    mapping["daii"]  = _C["da"] + _VS["ii"]
    mapping["dae"]   = _C["da"] + _VS["e"]
    mapping["dao"]   = _C["da"] + _VS["o"]
    mapping["dau"]   = _C["da"] + _VS["u"]
    mapping["dauu"]  = _C["da"] + _VS["uu"]

    # "dhai", "dhaii", etc
    mapping["dhai"]  = _C["dha"] + _VS["ai"]
    mapping["dhaii"] = _C["dha"] + _VS["ii"]
    mapping["dhae"]  = _C["dha"] + _VS["e"]
    mapping["dhao"]  = _C["dha"] + _VS["o"]
    mapping["dhau"]  = _C["dha"] + _VS["u"]
    mapping["dhauu"] = _C["dha"] + _VS["uu"]

    # "shai", "shaii", etc
    mapping["shai"]  = _C["sha"] + _VS["ai"]
    mapping["shaii"] = _C["sha"] + _VS["ii"]
    mapping["shae"]  = _C["sha"] + _VS["e"]
    mapping["shao"]  = _C["sha"] + _VS["o"]
    mapping["shau"]  = _C["sha"] + _VS["u"]

    # "thai", "thaii", etc
    mapping["thai"]  = _C["tha"] + _VS["ai"]
    mapping["thaii"] = _C["tha"] + _VS["ii"]
    mapping["thae"]  = _C["tha"] + _VS["e"]
    mapping["thao"]  = _C["tha"] + _VS["o"]
    mapping["thau"]  = _C["tha"] + _VS["u"]
    mapping["thauu"] = _C["tha"] + _VS["uu"]

    return mapping


# Pre-built mapping — importable as a dict
PHONETIC_TO_BRAHMI = _build_mapping()


def phonetic_to_brahmi(phonetic_label: str) -> str:
    """
    Convert a phonetic folder name to Brahmi Unicode text.

    Args:
        phonetic_label: e.g. "ka", "kaa", "bhi", "sha"

    Returns:
        Brahmi Unicode string, or the original label if not found.
    """
    return PHONETIC_TO_BRAHMI.get(phonetic_label, phonetic_label)


# ---------------------------------------------------------------------------
# Self-test: verify all 287 Capstone folders are mapped
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Try to find the Capstone dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    capstone_path = os.path.join(
        project_dir,
        "Capstone_Brahmi_Inscriptions", "OCR", "OCR_Dataset",
        "RecognizerDataset_150_210"
    )

    if os.path.isdir(capstone_path):
        folders = sorted(os.listdir(capstone_path))
        mapped = 0
        unmapped = []
        for f in folders:
            if f in PHONETIC_TO_BRAHMI:
                mapped += 1
            else:
                unmapped.append(f)
        print(f"Mapped: {mapped}/{len(folders)}")
        if unmapped:
            print(f"UNMAPPED: {unmapped}")
        else:
            print("All folders mapped ✓")
    else:
        print("Capstone dataset not found, showing sample mappings:")

    # Print sample mappings
    samples = ["ka", "kaa", "ki", "kii", "ku", "kuu", "ke", "ko",
               "ga", "bha", "sha", "tha", "dha", "na", "ma", "ya", "ra"]
    for s in samples:
        result = phonetic_to_brahmi(s)
        codepoints = " ".join(f"U+{ord(c):05X}" for c in result)
        print(f"  {s:10s} → {result}  ({codepoints})")
