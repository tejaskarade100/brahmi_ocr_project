"""
dataset/build_targets.py
========================

Analyzes the map.json dataset mapping and the current physical files on disk 
to generate a deterministic generation manifest (targets_manifest.csv).

This calculates exactly how many synthetic images are needed to reach the 
target quota per category:
- fixed classes (default: 100)
- words (default: 10000)
- phrases (default: 10000)
- sentences_multiline (default: 8000)

Styles are split across:
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

def count_images_in_folder_mixed(abs_folder: str) -> dict:
    counts = {"word": 0, "phrase": 0, "sentence": 0, "multiline": 0}
    json_path = os.path.join(abs_folder, "labels.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("entries", data.get("items", []))
            for item in items:
                file_ref = str(item.get("file", "")).strip()
                if not file_ref:
                    continue
                image_path = os.path.join(abs_folder, file_ref.replace("/", os.sep))
                if not os.path.isfile(image_path) or not _is_image_file(image_path):
                    continue
                seq_type = item.get("sequence_type")
                if not seq_type:
                    words = len([w for w in item.get("text_brahmi", "").split() if w])
                    if words <= 1: seq_type = "word"
                    elif words <= 4: seq_type = "phrase"
                    else: seq_type = "sentence"
                if seq_type in counts:
                    counts[seq_type] += 1
        except Exception:
            pass
    return counts

def count_images_in_folder_fixed(abs_folder: str) -> int:
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
    parser.add_argument("--target_fixed", type=int, default=100, help="Target images per fixed class")
    parser.add_argument("--target_words", type=int, default=10000, help="Target mixed word images")
    parser.add_argument("--target_phrases", type=int, default=10000, help="Target mixed phrase images")
    parser.add_argument("--target_sentences", type=int, default=8000, help="Target mixed sentence/multiline images total")
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

    def add_row(folder, label, entry_type, seq_type, current, target):
        nonlocal total_existing, total_needed
        need_gen = max(0, target - current)
        
        if need_gen > 0:
            style_clean = round(0.40 * need_gen)
            style_manuscript = round(0.35 * need_gen)
            style_stone = need_gen - style_clean - style_manuscript
        else:
            style_clean = style_manuscript = style_stone = 0

        total_existing += current
        total_needed += need_gen

        rows.append({
            "folder": folder,
            "label_text": label,
            "entry_type": entry_type,
            "sequence_type": seq_type,
            "current_count": current,
            "target_count": target,
            "need_generate": need_gen,
            "style_clean": style_clean,
            "style_manuscript": style_manuscript,
            "style_stone": style_stone
        })

    for entry in entries:
        folder_rel = entry["folder"].replace("/", os.sep)
        abs_folder = os.path.join(data_dir, folder_rel)
        label_text = entry["label_text"]

        if label_text == "MIXED":
            counts = count_images_in_folder_mixed(abs_folder)
            add_row(folder_rel, "MIXED", "mixed", "word", counts.get("word", 0), args.target_words)
            add_row(folder_rel, "MIXED", "mixed", "phrase", counts.get("phrase", 0), args.target_phrases)
            
            # Split sentences target between sentence and multiline
            sent_target = args.target_sentences // 2
            multi_target = args.target_sentences - sent_target
            
            add_row(folder_rel, "MIXED", "mixed", "sentence", counts.get("sentence", 0), sent_target)
            add_row(folder_rel, "MIXED", "mixed", "multiline", counts.get("multiline", 0), multi_target)
        else:
            current = count_images_in_folder_fixed(abs_folder)
            add_row(folder_rel, label_text, "fixed_class", "characters_ngrams", current, args.target_fixed)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "folder", "label_text", "entry_type", "sequence_type", "current_count", 
            "target_count", "need_generate", "style_clean", "style_manuscript", "style_stone"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n--- Balancer Summary ---")
    print(f"Total existing images : {total_existing}")
    print(f"Total images to build : {total_needed}")
    print(f"Manifest written to   : {os.path.relpath(out_path)}")

if __name__ == "__main__":
    main()
