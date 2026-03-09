"""
dataset/validate_dataset.py
===========================

Validate a map.json-driven Brahmi dataset before training.

Checks:
1) map.json structure and mapped folder coverage
2) sample collection from fixed classes + MIXED manifests
3) optional image integrity check
4) composition check against target ratios (default 60/25/15)
5) split simulation summary (train/val/test)

Usage:
    python dataset/validate_dataset.py --data_dir dataset
    python dataset/validate_dataset.py --data_dir dataset --json_out report.json
    python dataset/validate_dataset.py --data_dir dataset --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAP_META_KEYS = {"char", "latin", "unicode", "folder", "desc"}
MIXED_VALUE = "MIXED"
MIXED_LABEL_FILES = (
    "labels.json",
    "labels.txt",
    "labels.tsv",
    "labels.csv",
    "annotations.txt",
    "annotations.tsv",
    "annotations.csv",
)


@dataclass(frozen=True)
class MapEntry:
    folder: str
    label_text: str
    latin: str
    path_key: str


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    label_text: str
    source_folder: str
    source_type: str  # fixed_class | mixed_text


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _iter_image_files(folder_path: str) -> Iterable[str]:
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            if _is_image_file(full):
                yield os.path.normpath(full)


def _flatten_map_entries(node, path_prefix: str = "") -> List[MapEntry]:
    entries: List[MapEntry] = []
    if isinstance(node, dict):
        if "folder" in node and "char" in node:
            entries.append(
                MapEntry(
                    folder=str(node["folder"]),
                    label_text=str(node["char"]),
                    latin=str(node.get("latin", "")),
                    path_key=path_prefix,
                )
            )
        for key, value in node.items():
            if key in MAP_META_KEYS:
                continue
            child = f"{path_prefix}/{key}" if path_prefix else str(key)
            entries.extend(_flatten_map_entries(value, child))
    return entries


def load_map_entries(map_path: str) -> List[MapEntry]:
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _flatten_map_entries(data)


def _read_label_lines(label_file: str) -> Iterable[Tuple[str, str]]:
    ext = Path(label_file).suffix.lower()
    if ext == ".json":
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("entries", data.get("items", []))
        for item in items:
            image_ref = str(item.get("file", "")).strip()
            label_text = str(item.get("text_brahmi", "")).strip()
            if image_ref and label_text:
                yield image_ref, label_text
        return

    if ext == ".csv":
        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_ref = row[0].strip()
                label_text = ",".join(row[1:]).strip()
                if image_ref and label_text:
                    yield image_ref, label_text
        return

    with open(label_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                image_ref, label_text = line.split("\t", maxsplit=1)
            elif "," in line:
                image_ref, label_text = line.split(",", maxsplit=1)
            else:
                continue
            image_ref = image_ref.strip()
            label_text = label_text.strip()
            if image_ref and label_text:
                yield image_ref, label_text


def _resolve_mixed_samples(folder_abs: str, folder_rel: str) -> Tuple[List[SampleRecord], List[str]]:
    warnings: List[str] = []
    label_file = None
    for name in MIXED_LABEL_FILES:
        candidate = os.path.join(folder_abs, name)
        if os.path.isfile(candidate):
            label_file = candidate
            break

    if label_file is None:
        return [], [f"MIXED folder missing labels file: {folder_rel}"]

    records: List[SampleRecord] = []
    seen = set()
    for image_ref, label_text in _read_label_lines(label_file):
        image_path = image_ref
        if not os.path.isabs(image_path):
            image_path = os.path.join(folder_abs, image_ref.replace("/", os.sep))
        image_path = os.path.normpath(image_path)

        if not os.path.isfile(image_path):
            warnings.append(f"Missing image referenced in {folder_rel}: {image_ref}")
            continue
        if not _is_image_file(image_path):
            continue
        if image_path in seen:
            continue
        seen.add(image_path)
        records.append(
            SampleRecord(
                image_path=image_path,
                label_text=label_text,
                source_folder=folder_rel,
                source_type="mixed_text",
            )
        )

    return records, warnings


def collect_samples(data_dir: str, map_filename: str = "map.json"):
    map_path = os.path.join(data_dir, map_filename)
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"map not found: {map_path}")

    entries = load_map_entries(map_path)
    records: List[SampleRecord] = []
    warnings: List[str] = []
    seen = set()
    class_counts: Dict[str, int] = {}
    missing_folders: List[str] = []
    empty_folders: List[str] = []

    for entry in entries:
        rel = entry.folder.replace("/", os.sep)
        abs_path = os.path.join(data_dir, rel)

        if not os.path.isdir(abs_path):
            missing_folders.append(rel)
            class_counts[rel] = 0
            continue

        if entry.label_text.upper() == MIXED_VALUE:
            mixed_records, mix_warn = _resolve_mixed_samples(abs_path, rel)
            warnings.extend(mix_warn)
            class_counts[rel] = len(mixed_records)
            for rec in mixed_records:
                if rec.image_path in seen:
                    continue
                seen.add(rec.image_path)
                records.append(rec)
            if len(mixed_records) == 0:
                empty_folders.append(rel)
            continue

        count = 0
        for image_path in _iter_image_files(abs_path):
            if image_path in seen:
                continue
            seen.add(image_path)
            records.append(
                SampleRecord(
                    image_path=image_path,
                    label_text=entry.label_text,
                    source_folder=rel,
                    source_type="fixed_class",
                )
            )
            count += 1

        class_counts[rel] = count
        if count == 0:
            empty_folders.append(rel)

    return entries, records, class_counts, missing_folders, empty_folders, warnings


def classify_text(record: SampleRecord) -> str:
    if record.source_type == "fixed_class":
        return "characters_ngrams"

    words = [w for w in record.label_text.split(" ") if w]
    wc = len(words)
    if wc <= 1:
        return "words"
    if wc <= 4:
        return "phrases"
    return "long_sentences"


def summarize_records(records: Sequence[SampleRecord]) -> Dict:
    category_counts = Counter()
    char_len_hist = Counter()
    word_count_hist = Counter()

    for rec in records:
        cat = classify_text(rec)
        category_counts[cat] += 1

        text = rec.label_text
        char_len_hist[len(text.replace(" ", ""))] += 1
        word_count_hist[len([w for w in text.split(" ") if w])] += 1

    phrases_plus_sentences = category_counts.get("phrases", 0) + category_counts.get(
        "long_sentences", 0
    )

    return {
        "total_samples": len(records),
        "unique_labels": len({r.label_text for r in records}),
        "category_counts": {
            "characters_ngrams": int(category_counts.get("characters_ngrams", 0)),
            "words": int(category_counts.get("words", 0)),
            "phrases": int(category_counts.get("phrases", 0)),
            "long_sentences": int(category_counts.get("long_sentences", 0)),
            "sentences_phrases": int(phrases_plus_sentences),
        },
        "char_length_histogram": dict(sorted((int(k), int(v)) for k, v in char_len_hist.items())),
        "word_count_histogram": dict(sorted((int(k), int(v)) for k, v in word_count_hist.items())),
        "character_inventory_size": len({ch for r in records for ch in r.label_text if ch}),
    }


def check_image_integrity(records: Sequence[SampleRecord]) -> List[str]:
    bad: List[str] = []
    for rec in records:
        try:
            with Image.open(rec.image_path) as img:
                img.verify()
        except Exception:
            bad.append(rec.image_path)
    return bad


def split_records(
    records: Sequence[SampleRecord], ratios: Sequence[float], seed: int
) -> Dict[str, List[SampleRecord]]:
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to > 0")
    train_ratio, val_ratio, test_ratio = (
        train_ratio / total,
        val_ratio / total,
        test_ratio / total,
    )

    data = list(records)
    rng = random.Random(seed)
    rng.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_set = data[:n_train]
    val_set = data[n_train : n_train + n_val]
    test_set = data[n_train + n_val :]
    return {"train": train_set, "val": val_set, "test": test_set}


def evaluate_target_ratio(summary: Dict, target: Dict[str, float], tolerance: float) -> Dict:
    total = max(int(summary["total_samples"]), 1)
    counts = summary["category_counts"]
    actual = {
        "characters_ngrams": counts.get("characters_ngrams", 0) / total,
        "words": counts.get("words", 0) / total,
        "sentences_phrases": counts.get("sentences_phrases", 0) / total,
    }
    checks = {}
    for key, target_ratio in target.items():
        diff = abs(actual[key] - target_ratio)
        checks[key] = {
            "target_ratio": target_ratio,
            "actual_ratio": actual[key],
            "difference": diff,
            "within_tolerance": diff <= tolerance,
        }
    return {"actual": actual, "checks": checks}


def parse_args():
    p = argparse.ArgumentParser(description="Validate map.json-driven Brahmi dataset")
    p.add_argument("--data_dir", type=str, default="dataset", help="Dataset root directory")
    p.add_argument("--map_file", type=str, default="map.json", help="Map filename under data_dir")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_char_ngrams", type=float, default=0.60)
    p.add_argument("--target_words", type=float, default=0.25)
    p.add_argument("--target_sentences_phrases", type=float, default=0.15)
    p.add_argument("--ratio_tolerance", type=float, default=0.05)
    p.add_argument("--skip_image_check", action="store_true", help="Skip PIL verify for each image")
    p.add_argument("--json_out", type=str, default="", help="Optional report JSON output path")
    p.add_argument("--strict", action="store_true", help="Exit with code 1 on warnings or ratio mismatch")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    errors: List[str] = []
    warnings: List[str] = []

    try:
        entries, records, class_counts, missing_folders, empty_folders, gather_warnings = collect_samples(
            data_dir, map_filename=args.map_file
        )
        warnings.extend(gather_warnings)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    if not entries:
        errors.append("No map entries found in map.json")
    if not records:
        errors.append("No samples discovered from mapped folders")

    if missing_folders:
        warnings.append(f"Mapped folders missing on disk: {len(missing_folders)}")
    if empty_folders:
        warnings.append(f"Mapped folders with zero images: {len(empty_folders)}")

    integrity_bad = []
    if not args.skip_image_check and records:
        integrity_bad = check_image_integrity(records)
        if integrity_bad:
            warnings.append(f"Corrupted/unreadable images: {len(integrity_bad)}")

    summary = summarize_records(records)
    split_sets = split_records(records, (args.train_ratio, args.val_ratio, args.test_ratio), args.seed)
    split_summary = {k: summarize_records(v) for k, v in split_sets.items()}

    ratio_eval = evaluate_target_ratio(
        summary=summary,
        target={
            "characters_ngrams": args.target_char_ngrams,
            "words": args.target_words,
            "sentences_phrases": args.target_sentences_phrases,
        },
        tolerance=args.ratio_tolerance,
    )

    ratio_ok = all(v["within_tolerance"] for v in ratio_eval["checks"].values())
    if not ratio_ok:
        warnings.append("Dataset composition is outside target tolerance")

    basename_counts = Counter(os.path.basename(r.image_path) for r in records)
    dup_basenames = [(k, v) for k, v in basename_counts.items() if v > 1]

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_classes = sorted_classes[:20]
    bottom_classes = sorted(class_counts.items(), key=lambda x: x[1])[:20]

    report = {
        "data_dir": data_dir,
        "map_file": args.map_file,
        "mapped_class_count": len(entries),
        "mapped_classes_with_images": sum(1 for _, c in class_counts.items() if c > 0),
        "mapped_classes_missing_folder": len(missing_folders),
        "mapped_classes_empty": len(empty_folders),
        "total_image_samples": len(records),
        "summary": summary,
        "split_summary": split_summary,
        "ratio_evaluation": ratio_eval,
        "top_classes_by_count": top_classes,
        "bottom_classes_by_count": bottom_classes,
        "duplicate_basenames_count": len(dup_basenames),
        "duplicate_basenames_top20": sorted(dup_basenames, key=lambda x: x[1], reverse=True)[:20],
        "errors": errors,
        "warnings": warnings,
    }

    print("\n=== Dataset Validation Report ===")
    print(f"Data dir                 : {report['data_dir']}")
    print(f"Mapped classes           : {report['mapped_class_count']}")
    print(f"Mapped with images       : {report['mapped_classes_with_images']}")
    print(f"Mapped missing folder    : {report['mapped_classes_missing_folder']}")
    print(f"Mapped empty             : {report['mapped_classes_empty']}")
    print(f"Total samples            : {report['total_image_samples']}")

    cc = summary["category_counts"]
    total = max(summary["total_samples"], 1)
    print(
        "Composition              : "
        f"Chars/NGrams={100*cc['characters_ngrams']/total:.2f}% | "
        f"Words={100*cc['words']/total:.2f}% | "
        f"Sentences+Phrases={100*cc['sentences_phrases']/total:.2f}%"
    )

    print("\nTarget ratio check (with tolerance):")
    for key, data in ratio_eval["checks"].items():
        ok = "OK" if data["within_tolerance"] else "OUT"
        print(
            f"  {key:20s} {ok:3s} | "
            f"target={data['target_ratio']:.3f} "
            f"actual={data['actual_ratio']:.3f} "
            f"diff={data['difference']:.3f}"
        )

    if integrity_bad:
        print(f"\nCorrupted images         : {len(integrity_bad)}")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  - {w}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report written to   : {out_path}")

    if errors:
        return 2
    if args.strict and (warnings or not ratio_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
