"""
dataset/generate_synthetic.py — Synthetic Brahmi Dataset Generator
===================================================================

Generates synthetic images of Brahmi script text using the NotoSansBrahmi font.
Produces single characters, words, and phrases with random augmentations
(rotation, blur, noise, contrast, textured backgrounds) and writes them
to dataset/images/ with a labels.txt mapping and train/val/test splits.

USAGE:
    python dataset/generate_synthetic.py                          # defaults: 5k chars, 10k words, 5k phrases
    python dataset/generate_synthetic.py --num_chars 10 --num_words 10 --num_phrases 5  # small test run
"""

import argparse
import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


# ---------------------------------------------------------------------------
# Brahmi Unicode characters  (U+11000 – U+1107F)
# ---------------------------------------------------------------------------
# Independent vowels
BRAHMI_VOWELS = [chr(c) for c in range(0x11005, 0x11013)]

# Consonants
BRAHMI_CONSONANTS = [chr(c) for c in range(0x11013, 0x11038)]

# Dependent vowel signs (combine with consonants)
BRAHMI_VOWEL_SIGNS = [chr(c) for c in range(0x11038, 0x11047)]

# Brahmi digits
BRAHMI_DIGITS = [chr(c) for c in range(0x11052, 0x1106A)]

# Pool of base characters used to form text samples
BRAHMI_BASE_CHARS = BRAHMI_VOWELS + BRAHMI_CONSONANTS

# Pool of optional combining marks that can follow a consonant
BRAHMI_COMBINERS = BRAHMI_VOWEL_SIGNS


# ---------------------------------------------------------------------------
# Random text generators
# ---------------------------------------------------------------------------

def _random_char() -> str:
    """Return a single random Brahmi character, optionally with a vowel sign."""
    ch = random.choice(BRAHMI_BASE_CHARS)
    # 40 % chance to append a vowel sign after a consonant
    if ch in BRAHMI_CONSONANTS and random.random() < 0.4 and BRAHMI_COMBINERS:
        ch += random.choice(BRAHMI_COMBINERS)
    return ch


def generate_single_char() -> str:
    """Generate a single Brahmi character (possibly with combining mark)."""
    return _random_char()


def generate_word(min_len: int = 2, max_len: int = 6) -> str:
    """Generate a Brahmi word of *min_len* – *max_len* characters."""
    length = random.randint(min_len, max_len)
    return "".join(_random_char() for _ in range(length))


def generate_phrase(min_len: int = 5, max_len: int = 15) -> str:
    """Generate a short Brahmi phrase (space-separated words)."""
    target_chars = random.randint(min_len, max_len)
    words = []
    total = 0
    while total < target_chars:
        w = generate_word(2, 4)
        words.append(w)
        total += len(w)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Background generators  (procedural — no external texture files)
# ---------------------------------------------------------------------------

def _bg_plain_white(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), (255, 255, 255))


def _bg_parchment(w: int, h: int) -> Image.Image:
    """Yellowish-brown parchment tint with subtle noise."""
    arr = np.full((h, w, 3), [235, 220, 190], dtype=np.uint8)
    noise = np.random.randint(-12, 13, (h, w, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _bg_stone(w: int, h: int) -> Image.Image:
    """Grey stone-like texture with random speckles."""
    base = random.randint(160, 200)
    arr = np.full((h, w, 3), base, dtype=np.uint8)
    noise = np.random.randint(-25, 26, (h, w, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _bg_paper(w: int, h: int) -> Image.Image:
    """Slightly off-white paper."""
    arr = np.full((h, w, 3), [245, 242, 238], dtype=np.uint8)
    noise = np.random.randint(-6, 7, (h, w, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


BG_GENERATORS = [_bg_plain_white, _bg_parchment, _bg_stone, _bg_paper]


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def apply_rotation(img: Image.Image) -> Image.Image:
    """Random rotation ±15 degrees with white fill."""
    angle = random.uniform(-15, 15)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255))


def apply_gaussian_blur(img: Image.Image) -> Image.Image:
    """Random Gaussian blur (radius 0.5 – 1.5)."""
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_salt_pepper_noise(img: Image.Image, amount: float = 0.02) -> Image.Image:
    """Add salt-and-pepper noise."""
    arr = np.array(img)
    total_pixels = arr.shape[0] * arr.shape[1]
    # salt
    num_salt = int(total_pixels * amount / 2)
    coords_y = np.random.randint(0, arr.shape[0], num_salt)
    coords_x = np.random.randint(0, arr.shape[1], num_salt)
    arr[coords_y, coords_x] = 255
    # pepper
    num_pepper = int(total_pixels * amount / 2)
    coords_y = np.random.randint(0, arr.shape[0], num_pepper)
    coords_x = np.random.randint(0, arr.shape[1], num_pepper)
    arr[coords_y, coords_x] = 0
    return Image.fromarray(arr)


def apply_contrast_change(img: Image.Image) -> Image.Image:
    """Random contrast adjustment (0.7 – 1.3)."""
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(img).enhance(factor)


def augment(img: Image.Image) -> Image.Image:
    """Apply a random subset of augmentations."""
    if random.random() < 0.5:
        img = apply_rotation(img)
    if random.random() < 0.4:
        img = apply_gaussian_blur(img)
    if random.random() < 0.3:
        img = apply_salt_pepper_noise(img)
    if random.random() < 0.4:
        img = apply_contrast_change(img)
    return img


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def render_text_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    img_width: int = 384,
    img_height: int = 384,
) -> Image.Image:
    """
    Render *text* onto a random background, apply augmentations, and return
    an RGB PIL Image of size (img_width × img_height).
    """
    # --- choose a background ---
    bg_fn = random.choice(BG_GENERATORS)
    img = bg_fn(img_width, img_height)
    draw = ImageDraw.Draw(img)

    # --- choose a dark ink colour ---
    r = random.randint(0, 60)
    g = random.randint(0, 60)
    b = random.randint(0, 60)
    ink = (r, g, b)

    # --- measure text and centre it ---
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # If the text is wider than the image, shrink font dynamically
    effective_font = font
    if tw > img_width - 20 or th > img_height - 20:
        scale = min((img_width - 20) / max(tw, 1), (img_height - 20) / max(th, 1))
        new_size = max(int(font.size * scale), 12)
        try:
            effective_font = ImageFont.truetype(font.path, new_size)
        except Exception:
            effective_font = font
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=effective_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

    x = (img_width - tw) // 2
    y = (img_height - th) // 2
    draw.text((x, y), text, font=effective_font, fill=ink)

    # --- augment ---
    img = augment(img)

    return img


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    font_path: str,
    output_dir: str,
    num_chars: int = 5000,
    num_words: int = 10000,
    num_phrases: int = 5000,
    img_size: int = 384,
    font_size: int = 64,
    seed: int = 42,
):
    """Generate the full synthetic dataset and write labels + splits."""
    random.seed(seed)
    np.random.seed(seed)

    images_dir = os.path.join(output_dir, "images")
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # Load the Brahmi font
    font = ImageFont.truetype(font_path, font_size)

    # Build sample list: (generator_fn, count, font_size_override)
    samples_plan = []

    # Single characters — use a larger font
    char_font = ImageFont.truetype(font_path, font_size + 32)
    for _ in range(num_chars):
        samples_plan.append(("char", generate_single_char, char_font))

    # Words
    for _ in range(num_words):
        samples_plan.append(("word", generate_word, font))

    # Phrases — use a smaller font to fit
    phrase_font = ImageFont.truetype(font_path, max(font_size - 16, 28))
    for _ in range(num_phrases):
        samples_plan.append(("phrase", generate_phrase, phrase_font))

    total = len(samples_plan)
    labels = []

    print(f"Generating {total} synthetic Brahmi images …")
    for idx, (stype, gen_fn, sfont) in enumerate(samples_plan):
        text = gen_fn()
        img = render_text_image(text, sfont, img_size, img_size)

        fname = f"img_{idx + 1:06d}.png"
        img.save(os.path.join(images_dir, fname))
        labels.append((fname, text))

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1}/{total}] generated")

    # --- Write labels.txt ---
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for fname, text in labels:
            f.write(f"{fname}\t{text}\n")
    print(f"Labels written to {labels_path}")

    # --- Write splits ---
    all_names = [fname for fname, _ in labels]
    random.shuffle(all_names)
    n = len(all_names)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_names = all_names[:n_train]
    val_names = all_names[n_train : n_train + n_val]
    test_names = all_names[n_train + n_val :]

    for split_name, split_list in [
        ("train.txt", train_names),
        ("val.txt", val_names),
        ("test.txt", test_names),
    ]:
        path = os.path.join(splits_dir, split_name)
        with open(path, "w", encoding="utf-8") as f:
            for name in split_list:
                f.write(name + "\n")
        print(f"  {split_name}: {len(split_list)} samples")

    print("Done ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Brahmi OCR dataset"
    )
    parser.add_argument(
        "--font",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "NotoSansBrahmi-Regular.ttf",
        ),
        help="Path to the NotoSansBrahmi-Regular.ttf font file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
        ),
        help="Root output directory (default: dataset/)",
    )
    parser.add_argument("--num_chars", type=int, default=5000)
    parser.add_argument("--num_words", type=int, default=10000)
    parser.add_argument("--num_phrases", type=int, default=5000)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--font_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        font_path=args.font,
        output_dir=args.output_dir,
        num_chars=args.num_chars,
        num_words=args.num_words,
        num_phrases=args.num_phrases,
        img_size=args.img_size,
        font_size=args.font_size,
        seed=args.seed,
    )
