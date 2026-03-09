"""
dataset/postcheck.py
====================

Runs perceptual deduplication and quota audits across the generated dataset.
 Ensures the data matches the manifest completely.
"""

import argparse
import csv
import json
import os
import sys

from PIL import Image

def build_phash(img: Image.Image, hash_size: int = 8) -> str:
    """Computes a simple deterministic perceptual hash."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = img.getpixel((col, row))
            pixel_right = img.getpixel((col + 1, row))
            diff.append(pixel_left > pixel_right)
    return hex(int("".join(["1" if b else "0" for b in diff]), 2))[2:]

def _is_image_file(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--manifest", type=str, default="dataset/reports/targets_manifest.csv")
    parser.add_argument("--skip_dedupe", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    manifest_path = os.path.abspath(args.manifest)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        targets = list(reader)

    print("--- Running Post-Generation Audit ---")
    
    total_duplicates_removed = 0
    quota_failures = 0

    for row in targets:
        folder_rel = row["folder"]
        target_count = int(row["target_count"])
        abs_folder = os.path.join(data_dir, folder_rel)
        
        if not os.path.exists(abs_folder):
            print(f"[FAIL] Missing target directory: {folder_rel}")
            quota_failures += 1
            continue

        images = []
        for root, _, files in os.walk(abs_folder):
            for name in files:
                if _is_image_file(name):
                    images.append(os.path.join(root, name))

        # 1. Perceptual Deduplication
        if not args.skip_dedupe and len(images) > 0:
            seen_hashes = set()
            todelete = []
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        phash = build_phash(img)
                    if phash in seen_hashes:
                        todelete.append(img_path)
                    else:
                        seen_hashes.add(phash)
                except Exception as e:
                    print(f"Read error on {img_path}: {e}")
                    
            for bad in todelete:
                try:
                    os.remove(bad)
                    images.remove(bad)
                    total_duplicates_removed += 1
                except Exception as e:
                    print(f"Could not remove {bad}")

        # 2. Quota Check
        current_count = len(images)
        if current_count < target_count * 0.95:  # Tolerance for dedupes
            print(f"[WARN] Quota underfilled: {folder_rel} (Has: {current_count}, Target: {target_count})")
            quota_failures += 1

    print(f"\\n--- Audit Complete ---")
    if not args.skip_dedupe:
        print(f"Duplicates pruned: {total_duplicates_removed}")
    print(f"Folders under quota: {quota_failures}")
    if quota_failures == 0:
        print("✅ Dataset generation successful and quotas met.")
    else:
        print("⚠️ Some classes did not meet the generation quotas.")

if __name__ == "__main__":
    main()
