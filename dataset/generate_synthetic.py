"""
dataset/generate_synthetic.py
=============================

Mass-generates synthetic Brahmi training images from a deterministic manifest.
Applies modular style pipelines: Clean, Manuscript, and Stone Inscription.
Automatically handles single-character generation and MIXED phrase generation.

Usage:
  python dataset/generate_synthetic.py --dry_run
  python dataset/generate_synthetic.py
"""

import argparse
import csv
import json
import math
import os
import random
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from scipy.ndimage import map_coordinates, gaussian_filter
except ImportError:
    print("WARNING: scipy not found. Elastic distortions will fall back to simple affine warps.")

try:
    import freetype
except ImportError:
    freetype = None

try:
    import uharfbuzz as hb
except ImportError:
    hb = None

FONT_PATH = "NotoSansBrahmi-Regular.ttf"
DEFAULT_SEED = 42
BRAHMI_DEPENDENT_SIGNS = {chr(cp) for cp in range(0x11038, 0x11046)}
BRAHMI_NON_START_CHARS = BRAHMI_DEPENDENT_SIGNS | {chr(0x11046), chr(0x1107F)}
_HB_RENDERER_CACHE: Dict[Tuple[str, int], "_HarfBuzzRenderer"] = {}
_WARNED_BASIC_LAYOUT = False


def _load_pillow_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    if hasattr(ImageFont, "Layout") and hasattr(ImageFont.Layout, "RAQM"):
        try:
            return ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
        except Exception:
            pass
    return ImageFont.truetype(font_path, font_size)


def _is_valid_mixed_token(token: str) -> bool:
    if not token or token == "MIXED":
        return False
    if any(ch.isspace() for ch in token):
        return False
    return token[0] not in BRAHMI_NON_START_CHARS


def build_mixed_token_pools(rows: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    consonant_pool: List[str] = []
    vowel_pool: List[str] = []
    seen_consonants = set()
    seen_vowels = set()

    for row in rows:
        token = row["label_text"].strip()
        folder = row["folder"].replace("\\", "/")
        if not _is_valid_mixed_token(token):
            continue
        if folder.startswith("2Consonants/"):
            if token not in seen_consonants:
                consonant_pool.append(token)
                seen_consonants.add(token)
        elif folder.startswith("1Vowels/"):
            if token not in seen_vowels:
                vowel_pool.append(token)
                seen_vowels.add(token)

    if not consonant_pool:
        consonant_pool = [
            "𑀓", "𑀔", "𑀕", "𑀖", "𑀗", "𑀘", "𑀙", "𑀚", "𑀛", "𑀜",
            "𑀝", "𑀞", "𑀟", "𑀠", "𑀡", "𑀢", "𑀣", "𑀤", "𑀥", "𑀦",
            "𑀧", "𑀨", "𑀩", "𑀪", "𑀫", "𑀬", "𑀭", "𑀮", "𑀯", "𑀰",
            "𑀱", "𑀲", "𑀳",
        ]

    return consonant_pool, vowel_pool


class _HarfBuzzRenderer:
    def __init__(self, font_path: str, font_size: int):
        self.font_path = font_path
        self.font_size = font_size
        with open(font_path, "rb") as f:
            font_data = f.read()
        self.face = freetype.Face(font_path)
        self.face.set_pixel_sizes(0, font_size)
        blob = hb.Blob(font_data)
        hb_face = hb.Face(blob, 0)
        self.hb_font = hb.Font(hb_face)
        hb.ot_font_set_funcs(self.hb_font)
        self.hb_font.scale = (self.face.size.x_ppem << 6, self.face.size.y_ppem << 6)
        self.hb_font.ppem = (self.face.size.x_ppem, self.face.size.y_ppem)
        self._load_flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL

    def _shape(self, text: str):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {"kern": True, "liga": True, "mark": True, "mkmk": True})
        return buf.glyph_infos, buf.glyph_positions

    def render(self, text: str, padding: int) -> Image.Image:
        infos, positions = self._shape(text)
        if not infos:
            return Image.new("RGB", (64, 64), color="white")

        x_cursor = 0
        y_cursor = 0
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for info, pos in zip(infos, positions):
            self.face.load_glyph(info.codepoint, self._load_flags)
            slot = self.face.glyph
            bmp = slot.bitmap
            if bmp.width > 0 and bmp.rows > 0:
                gx = (x_cursor + pos.x_offset) / 64.0 + slot.bitmap_left
                gy = (y_cursor + pos.y_offset) / 64.0 - slot.bitmap_top
                min_x = min(min_x, gx)
                min_y = min(min_y, gy)
                max_x = max(max_x, gx + bmp.width)
                max_y = max(max_y, gy + bmp.rows)
            x_cursor += pos.x_advance
            y_cursor += pos.y_advance

        if min_x == float("inf"):
            width = max(64, int(math.ceil((x_cursor / 64.0) + padding * 2)))
            height = max(64, int(math.ceil((self.face.size.height / 64.0) + padding * 2)))
            return Image.new("RGB", (width, height), color="white")

        width = max(64, int(math.ceil(max_x - min_x + padding * 2)))
        height = max(64, int(math.ceil(max_y - min_y + padding * 2)))
        offset_x = padding - min_x
        offset_y = padding - min_y

        canvas = np.full((height, width), 255, dtype=np.uint8)
        x_cursor = 0
        y_cursor = 0

        for info, pos in zip(infos, positions):
            self.face.load_glyph(info.codepoint, self._load_flags)
            slot = self.face.glyph
            bmp = slot.bitmap
            if bmp.width > 0 and bmp.rows > 0:
                glyph = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width)
                gx = int(round((x_cursor + pos.x_offset) / 64.0 + slot.bitmap_left + offset_x))
                gy = int(round((y_cursor + pos.y_offset) / 64.0 - slot.bitmap_top + offset_y))

                x0 = max(0, gx)
                y0 = max(0, gy)
                x1 = min(width, gx + bmp.width)
                y1 = min(height, gy + bmp.rows)

                if x0 < x1 and y0 < y1:
                    crop = glyph[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                    region = canvas[y0:y1, x0:x1]
                    canvas[y0:y1, x0:x1] = np.minimum(region, 255 - crop)

            x_cursor += pos.x_advance
            y_cursor += pos.y_advance

        return Image.fromarray(canvas, mode="L").convert("RGB")


def _get_hb_renderer(font_path: str, font_size: int) -> Optional[_HarfBuzzRenderer]:
    if hb is None or freetype is None:
        return None
    key = (os.path.abspath(font_path), font_size)
    renderer = _HB_RENDERER_CACHE.get(key)
    if renderer is None:
        renderer = _HarfBuzzRenderer(font_path, font_size)
        _HB_RENDERER_CACHE[key] = renderer
    return renderer


class StyleEngine:
    """Applies perceptual variations to base rendered text."""
    
    @staticmethod
    def _elastic_distortion(image: np.ndarray, alpha: float, sigma: float, rng: random.Random) -> np.ndarray:
        if 'map_coordinates' not in globals():
            return image
        shape = image.shape
        dx = gaussian_filter([rng.random() * 2 - 1 for _ in range(shape[0]*shape[1])], sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter([rng.random() * 2 - 1 for _ in range(shape[0]*shape[1])], sigma, mode="constant", cval=0) * alpha
        dx = np.array(list(dx)).reshape(shape)
        dy = np.array(list(dy)).reshape(shape)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        return distorted
        
    @staticmethod
    def _add_noise(image: np.ndarray, intensity: float, rng: random.Random) -> np.ndarray:
        noise = np.array([rng.randint(0, int(255 * intensity)) for _ in range(image.size)], dtype=np.uint8)
        noise = noise.reshape(image.shape)
        noisy = cv2.add(image, noise)
        return noisy

    @classmethod
    def apply_clean(cls, img: Image.Image, rng: random.Random) -> Image.Image:
        # 40%: Small rotation, mild shear, clean background
        angle = rng.uniform(-3, 3)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        return img

    @classmethod
    def apply_manuscript(cls, img: Image.Image, rng: random.Random) -> Image.Image:
        # 35%: Moderate rotation, blur, ink bleed
        angle = rng.uniform(-8, 8)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(240, 240, 230)) # Slight off-white paper
        
        np_img = np.array(img.convert('L'))
        
        # simulated ink bleed (erosion makes black text thicker)
        kernel = np.ones((2, 2), np.uint8)
        if rng.random() > 0.5:
            np_img = cv2.erode(np_img, kernel, iterations=1)
        
        np_img = cls._add_noise(np_img, 0.1, rng)
        img = Image.fromarray(np_img).convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.2)))
        return img

    @classmethod
    def apply_stone(cls, img: Image.Image, rng: random.Random) -> Image.Image:
        # 25%: Strong erosion, uneven lighting, rough texture
        angle = rng.uniform(-15, 15)
        bg_color = (rng.randint(150, 200), rng.randint(150, 200), rng.randint(150, 200)) # Stone greyish
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg_color)
        
        np_img = np.array(img.convert('L'))
        
        # Heavy distortion/erosion
        if 'map_coordinates' in globals():
            np_img = cls._elastic_distortion(np_img, alpha=rng.uniform(15, 30), sigma=rng.uniform(3, 5), rng=rng)
            
        kernel = np.ones((3, 3), np.uint8)
        if rng.random() > 0.5:
            np_img = cv2.dilate(np_img, kernel, iterations=1) # Faded/chipped text
        else:
            np_img = cv2.erode(np_img, kernel, iterations=1)  # Deep carved text

        np_img = cls._add_noise(np_img, 0.3, rng)
        
        # Uneven lighting (gradient)
        h, w = np_img.shape
        gradient = np.tile(np.linspace(1.0, rng.uniform(0.4, 0.8), w), (h, 1))
        np_img = np.clip(np_img * gradient, 0, 255).astype(np.uint8)

        img = Image.fromarray(np_img).convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.0)))
        return img


def render_base_text(text: str, font_path: str, font_size: int = 120, padding: int = 20) -> Image.Image:
    global _WARNED_BASIC_LAYOUT

    hb_renderer = _get_hb_renderer(font_path, font_size)
    if hb_renderer is not None:
        return hb_renderer.render(text, padding)

    try:
        font = _load_pillow_font(font_path, font_size)
    except IOError:
        raise RuntimeError(f"Cannot load font from {font_path}. Please ensure it exists.")
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize font renderer for {font_path}: {exc}") from exc

    if not _WARNED_BASIC_LAYOUT and any(ch in BRAHMI_DEPENDENT_SIGNS for ch in text):
        print(
            "WARNING: HarfBuzz shaping unavailable (install uharfbuzz + freetype-py). "
            "Dependent vowel signs may render detached."
        )
        _WARNED_BASIC_LAYOUT = True

    dummy_img = Image.new('RGB', (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    
    width = bbox[2] - bbox[0] + padding * 2
    height = bbox[3] - bbox[1] + padding * 2
    
    # Needs to be at least some realistic size
    width = max(width, 64)
    height = max(height, 64)
    
    img = Image.new('RGB', (int(width), int(height)), color="white")
    draw = ImageDraw.Draw(img)
    
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill="black")
    return img


def generate_mixed_phrase(
    consonant_pool: List[str],
    vowel_pool: List[str],
    rng: random.Random,
) -> str:
    """Generates 2-5 words from valid Brahmi syllable units."""
    num_words = rng.randint(2, 5)
    words = []
    for _ in range(num_words):
        word_len = rng.randint(2, 4)
        syllables: List[str] = []
        for idx in range(word_len):
            if idx == 0 and vowel_pool and rng.random() < 0.15:
                syllables.append(rng.choice(vowel_pool))
            elif consonant_pool and (not vowel_pool or rng.random() < 0.85):
                syllables.append(rng.choice(consonant_pool))
            elif vowel_pool:
                syllables.append(rng.choice(vowel_pool))
            else:
                syllables.append("𑀓")
        words.append("".join(syllables))
    return " ".join(words)


def process_manifest(manifest_path: str, data_dir: str, dry_run: bool = False, batch_limit: int = 100):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    consonant_pool, vowel_pool = build_mixed_token_pools(rows)
    print(
        f"MIXED token pools -> consonant_syllables={len(consonant_pool)}, "
        f"independent_vowels={len(vowel_pool)}"
    )

    total_generated = 0

    for row in rows:
        folder_rel = row["folder"]
        label_text = row["label_text"]
        need_generate = int(row["need_generate"])
        
        if dry_run:
            need_generate = min(batch_limit, need_generate)
            
        if need_generate <= 0:
            if not dry_run:
                print(f"Skipping {folder_rel} (Quota met)")
            continue

        counts = {
            "clean": int(row["style_clean"]),
            "manuscript": int(row["style_manuscript"]),
            "stone": int(row["style_stone"])
        }
        
        if dry_run:
            counts = {
                "clean": math.ceil(counts["clean"] / int(row["need_generate"]) * need_generate) if need_generate else 0,
                "manuscript": math.ceil(counts["manuscript"] / int(row["need_generate"]) * need_generate) if need_generate else 0,
                "stone": math.ceil(counts["stone"] / int(row["need_generate"]) * need_generate) if need_generate else 0,
            }

        out_dir = os.path.join(data_dir, folder_rel)
        os.makedirs(out_dir, exist_ok=True)
        
        folder_seed = DEFAULT_SEED + hash(folder_rel) % 1000000
        rng = random.Random(folder_seed)
        
        is_mixed = (label_text == "MIXED")
        json_entries = []

        print(f"Generating {need_generate} for {folder_rel} (Mixed: {is_mixed})")

        idx = 0
        for style, count in counts.items():
            for _ in range(count):
                if idx >= need_generate:
                    break
                    
                target_text = label_text
                if is_mixed:
                    target_text = generate_mixed_phrase(consonant_pool, vowel_pool, rng)
                
                # Render Base Image
                base_img = render_base_text(target_text, FONT_PATH, font_size=rng.randint(80, 140))
                
                # Apply Styles
                if style == "clean":
                    final_img = StyleEngine.apply_clean(base_img, rng)
                elif style == "manuscript":
                    final_img = StyleEngine.apply_manuscript(base_img, rng)
                else:
                    final_img = StyleEngine.apply_stone(base_img, rng)
                
                # Save optimized
                uid = uuid.UUID(int=rng.getrandbits(128)).hex[:8]
                filename = f"gen_{style}_{uid}.webp"
                save_path = os.path.join(out_dir, filename)
                
                final_img.save(save_path, "WEBP", quality=85)
                total_generated += 1
                
                if is_mixed:
                    json_entries.append({
                        "file": filename,
                        "text_brahmi": target_text,
                        "source": "synthetic",
                        "style": style,
                        "seed": folder_seed + idx
                    })
                
                idx += 1
                
                if dry_run and idx >= batch_limit:
                    break

        if is_mixed and json_entries:
            json_path = os.path.join(out_dir, "labels.json")
            existing_data = {"version": "1.0", "entries": []}
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        existing_data = json.load(jf)
                except Exception:
                    pass
            existing_data["entries"].extend(json_entries)
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(existing_data, jf, indent=2, ensure_ascii=False)

    print(f"\\nGeneration Complete. Generated {total_generated} images.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--manifest", type=str, default="dataset/reports/targets_manifest.csv", help="Targets manifest path")
    parser.add_argument("--dry_run", action="store_true", help="Only generate a few samples per class")
    parser.add_argument("--batch_limit", type=int, default=5, help="Limit per class if dry_run")
    args = parser.parse_args()

    process_manifest(args.manifest, args.data_dir, dry_run=args.dry_run, batch_limit=args.batch_limit)

if __name__ == "__main__":
    main()
