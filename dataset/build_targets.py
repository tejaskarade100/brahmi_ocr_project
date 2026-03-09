"""
dataset/build_targets.py
========================

Analyzes the map.json dataset mapping and the current physical files on disk 
to generate a deterministic generation manifest (targets_manifest.csv).

This calculates exactly how many synthetic images are needed to reach the 
target quota (default 1000) per class, split across 3 styles:
- Clean Font (40%)
- Manuscript (35%)
- Stone Inscription (25%)
"""

import argparse
import csv
import json
import os
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAP_META_KEYS = {"char", "latin", "unicode", "folder", "desc"}

def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS

def _flatten_map_entries(node, path_prefix: str = "") -> list:
    entries = []
    if isinstance(node, dict):
        if "folder" in node and "char" in node:
            entries.append({
                "folder": str(node["folder"]),
                "label_text": str(node["char"])
            })
        for key, value in node.items():
            if key in MAP_META_KEYS:
                continue
            child = f"{path_prefix}/{key}" if path_prefix else str(key)
            entries.extend(_flatten_map_entries(value, child))
    return entries

def load_map_entries(map_path: str) -> list:
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _flatten_map_entries(data)

def count_images_in_folder(abs_folder: str) -> int:
    if not os.path.exists(abs_folder):
        return 0
    count = 0
    for root, _, files in os.walk(abs_folder):
        for name in files:
            if _is_image_file(name):
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--map_file", type=str, default="map.json", help="Map file name")
    parser.add_argument("--target_count", type=int, default=1000, help="Target images per class")
    parser.add_argument("--out_csv", type=str, default="dataset/reports/targets_manifest.csv", help="Output manifest file path")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    map_path = os.path.join(data_dir, args.map_file)

    if not os.path.exists(map_path):
        print(f"Error: map file not found at {map_path}")
        return

    entries = load_map_entries(map_path)
    print(f"Loaded {len(entries)} mapped classes from {args.map_file}")

    out_path = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    total_existing = 0
    total_needed = 0

    for entry in entries:
        folder_rel = entry["folder"].replace("/", os.sep)
        abs_folder = os.path.join(data_dir, folder_rel)
        label_text = entry["label_text"]

        current_count = count_images_in_folder(abs_folder)
        need_generate = max(0, args.target_count - current_count)

        if need_generate > 0:
            style_clean = round(0.40 * need_generate)
            style_manuscript = round(0.35 * need_generate)
            style_stone = need_generate - style_clean - style_manuscript
        else:
            style_clean = style_manuscript = style_stone = 0

        total_existing += current_count
        total_needed += need_generate

        rows.append({
            "folder": folder_rel,
            "label_text": label_text,
            "current_count": current_count,
            "target_count": args.target_count,
            "need_generate": need_generate,
            "style_clean": style_clean,
            "style_manuscript": style_manuscript,
            "style_stone": style_stone
        })

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "folder", "label_text", "current_count", "target_count", "need_generate",
            "style_clean", "style_manuscript", "style_stone"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\\n--- Balancer Summary ---")
    print(f"Total existing images : {total_existing}")
    print(f"Total images to build : {total_needed}")
    print(f"Manifest written to   : {os.path.relpath(out_path)}")

if __name__ == "__main__":
    main()
