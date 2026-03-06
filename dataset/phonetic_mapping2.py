"""
dataset/phonetic_mapping2.py — Phonetic label → Brahmi Unicode mapping for BrahmiGAN
=====================================================================================

Maps the folder names used by the BrahmiGAN dataset to the correct
Brahmi Unicode character(s) (U+11000 – U+1107F).

The BrahmiGAN dataset uses standard IAST diacritics in its folder names 
(e.g., 'śa', 'ḍa', 'ṇa', 'ṭa') instead of the ASCII-only fallbacks used 
by Capstone ('sha', 'dda', 'nna', 'tta').

USAGE:
    from dataset.phonetic_mapping2 import phonetic_to_brahmi2
    brahmi_text = phonetic_to_brahmi2("śa")   # → "𑀰"
"""

# The BrahmiGAN dataset contains 23 core character classes
_BRAHMIGAN_MAPPING = {
    # Vowels
    "a":  "\U00011005",  # 𑀅
    "u":  "\U00011009",  # 𑀉
    
    # Consonants 
    "ka": "\U00011013",  # 𑀓
    "ga": "\U00011015",  # 𑀕
    "ca": "\U00011018",  # 𑀘
    "ja": "\U0001101A",  # 𑀚
    "jha": "\U0001101B", # 𑀛
    "ṭa": "\U0001101D",  # 𑀝  (retroflex ta)
    "ḍa": "\U0001101F",  # 𑀟  (retroflex da)
    "ṇa": "\U00011021",  # 𑀡  (retroflex na)
    "ta": "\U00011022",  # 𑀢
    "da": "\U00011024",  # 𑀤
    "na": "\U00011026",  # 𑀦
    "pa": "\U00011027",  # 𑀧
    "ba": "\U00011029",  # 𑀩
    "ma": "\U0001102B",  # 𑀫
    "ya": "\U0001102C",  # 𑀬
    "ra": "\U0001102D",  # 𑀭
    "la": "\U0001102E",  # 𑀮
    "va": "\U0001102F",  # 𑀯
    "śa": "\U00011030",  # 𑀰  (palatal sa)
    "sa": "\U00011032",  # 𑀲
    "ha": "\U00011033",  # 𑀳
}


def phonetic_to_brahmi2(phonetic_label: str) -> str:
    """
    Convert a BrahmiGAN phonetic folder name (with IAST diacritics) 
    to Brahmi Unicode text.

    Args:
        phonetic_label: e.g. "ka", "śa", "ṭa"

    Returns:
        Brahmi Unicode string, or the original label if not found.
    """
    return _BRAHMIGAN_MAPPING.get(phonetic_label, phonetic_label)


# ---------------------------------------------------------------------------
# Self-test: verify all 23 BrahmiGAN folders are mapped
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Try to find the BrahmiGAN dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    brahmigan_path = os.path.join(
        project_dir,
        "BrahmiGAN", "BrahmiGAN"
    )

    if os.path.isdir(brahmigan_path):
        folders = sorted(os.listdir(brahmigan_path))
        mapped = 0
        unmapped = []
        for f in folders:
            if f in _BRAHMIGAN_MAPPING:
                mapped += 1
            else:
                unmapped.append(f)
        print(f"Mapped: {mapped}/{len(folders)}")
        if unmapped:
            print(f"UNMAPPED: {unmapped}")
        else:
            print("All folders mapped ✓")
            
        print("\nMappings:")
        for s in folders:
            result = phonetic_to_brahmi2(s)
            codepoints = " ".join(f"U+{ord(c):05X}" for c in result)
            print(f"  {s:10s} → {result}  ({codepoints})")
    else:
        print("BrahmiGAN dataset not found at expected path.")
