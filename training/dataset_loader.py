"""
training/dataset_loader.py
==========================

Dataset utilities for Brahmi OCR training using `dataset/map.json` as the
single source of truth for labels and folder structure.
Adds runtime class bounding and a weighted sampler for length-balanced batches.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from utils.preprocess import letterbox_pil


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAP_METADATA_KEYS = {"char", "latin", "unicode", "folder", "desc"}
MIXED_LABEL_VALUE = "MIXED"
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
class SampleRecord:
    image_path: str
    label_text: str
    source_folder: str
    source_type: str  # fixed_class | mixed_text
    sequence_type: Optional[str] = None


@dataclass(frozen=True)
class MapEntry:
    folder: str
    label_text: str
    latin: str
    path_key: str


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def _iter_image_files(folder_path: str) -> Iterable[str]:
    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        for name in sorted(files):
            full_path = os.path.join(root, name)
            if _is_image_file(full_path):
                yield full_path


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
            if key in MAP_METADATA_KEYS:
                continue
            child_prefix = f"{path_prefix}/{key}" if path_prefix else str(key)
            entries.extend(_flatten_map_entries(value, child_prefix))

    return entries


def load_map_entries(map_path: str) -> List[MapEntry]:
    with open(map_path, "r", encoding="utf-8") as f:
        map_data = json.load(f)
    return _flatten_map_entries(map_data)


def _read_label_lines_enriched(label_file_path: str) -> Iterable[Tuple[str, str, Optional[str]]]:
    ext = os.path.splitext(label_file_path)[1].lower()
    
    if ext == ".json":
        with open(label_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("entries", data.get("items", []))
        for item in items:
            image_ref = item.get("file", "").strip()
            text = item.get("text_brahmi", "").strip()
            seq_type = item.get("sequence_type", "").strip() or None
            if image_ref and text:
                yield image_ref, text, seq_type
        return

    delimiter = ","
    if ext in {".txt", ".tsv"}:
        delimiter = "\t"

    with open(label_file_path, "r", encoding="utf-8") as f:
        if delimiter == ",":
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                image_rel = row[0].strip()
                text = ",".join(row[1:]).strip()
                if image_rel and text:
                    yield image_rel, text, None
            return

        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                image_rel, text = line.split("\t", maxsplit=1)
            elif "," in line:
                image_rel, text = line.split(",", maxsplit=1)
            else:
                continue
            if image_rel.strip() and text.strip():
                yield image_rel.strip(), text.strip(), None


def _resolve_mixed_labels(folder_abs: str, folder_rel: str) -> List[SampleRecord]:
    label_file = None
    for candidate in MIXED_LABEL_FILES:
        path = os.path.join(folder_abs, candidate)
        if os.path.isfile(path):
            label_file = path
            break

    if label_file is None:
        return []

    samples: List[SampleRecord] = []
    seen = set()
    for image_ref, text, seq_type in _read_label_lines_enriched(label_file):
        image_path = image_ref
        if not os.path.isabs(image_path):
            image_path = os.path.join(folder_abs, image_ref.replace("/", os.sep))
        image_path = os.path.normpath(image_path)

        if not os.path.isfile(image_path) or not _is_image_file(image_path) or image_path in seen:
            continue
        seen.add(image_path)

        samples.append(
            SampleRecord(
                image_path=image_path,
                label_text=text,
                source_folder=folder_rel,
                source_type="mixed_text",
                sequence_type=seq_type
            )
        )
    return samples


def load_capped_samples(
    data_dir: str, 
    map_filename: str = "map.json",
    max_fixed_per_class: int = 100,
    max_words: int = 10000,
    max_phrases: int = 10000,
    max_long_sentences: int = 8000,
    seed: int = 42
) -> List[SampleRecord]:
    map_path = os.path.join(data_dir, map_filename)
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"map file not found: {map_path}")

    entries = load_map_entries(map_path)
    
    fixed_samples_by_folder = {}
    mixed_samples_by_category = {
        "words": [],
        "phrases": [],
        "long_sentences": []
    }
    
    seen_image_paths = set()
    rng = random.Random(seed)

    # Gather data into buckets
    for entry in entries:
        folder_rel = entry.folder.replace("/", os.sep)
        folder_abs = os.path.join(data_dir, folder_rel)

        if not os.path.isdir(folder_abs):
            continue

        label = entry.label_text
        if label.upper() == MIXED_LABEL_VALUE:
            mixed_samples = _resolve_mixed_labels(folder_abs, folder_rel)
            for rec in mixed_samples:
                if rec.image_path in seen_image_paths:
                    continue
                seen_image_paths.add(rec.image_path)
                
                cat = _classify_text_shape(rec)
                if cat in mixed_samples_by_category:
                    mixed_samples_by_category[cat].append(rec)
            continue

        # Fixed category
        if folder_rel not in fixed_samples_by_folder:
            fixed_samples_by_folder[folder_rel] = []
            
        for image_path in _iter_image_files(folder_abs):
            norm_path = os.path.normpath(image_path)
            if norm_path in seen_image_paths:
                continue
            seen_image_paths.add(norm_path)
            fixed_samples_by_folder[folder_rel].append(
                SampleRecord(
                    image_path=norm_path,
                    label_text=label,
                    source_folder=folder_rel,
                    source_type="fixed_class",
                )
            )

    final_samples = []
    
    # Cap Fixed
    for folder, smps in fixed_samples_by_folder.items():
        rng.shuffle(smps)
        final_samples.extend(smps[:max_fixed_per_class])
        
    # Cap Mixed
    for cat, cap in zip(["words", "phrases", "long_sentences"], [max_words, max_phrases, max_long_sentences]):
        smps = mixed_samples_by_category[cat]
        rng.shuffle(smps)
        final_samples.extend(smps[:cap])

    final_samples.sort(key=lambda s: s.image_path)
    return final_samples


def _normalize_ratios(split_ratios: Sequence[float]) -> Tuple[float, float, float]:
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must be 3 values: train, val, test")
    train_ratio, val_ratio, test_ratio = split_ratios
    total = train_ratio + val_ratio + test_ratio
    return train_ratio / total, val_ratio / total, test_ratio / total


def split_samples(
    samples: Sequence[SampleRecord],
    split: str,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> List[SampleRecord]:
    split = split.lower().strip()
    if split == "all": return list(samples)

    train_ratio, val_ratio, _ = _normalize_ratios(split_ratios)
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if split == "train": return shuffled[:n_train]
    if split == "val": return shuffled[n_train : n_train + n_val]
    return shuffled[n_train + n_val :]


def _classify_text_shape(record: SampleRecord) -> str:
    if record.source_type == "fixed_class":
        return "characters_ngrams"

    if record.sequence_type:
        val = record.sequence_type
        if val in {"sentence", "multiline"}:
            return "long_sentences"
        if val in {"word", "phrase"}:
            return val + "s"

    words = [w for w in record.label_text.split(" ") if w]
    word_count = len(words)
    if word_count <= 1: return "words"
    if word_count <= 4: return "phrases"
    return "long_sentences"


def summarize_samples(samples: Sequence[SampleRecord]) -> Dict:
    category_counts = {
        "characters_ngrams": 0,
        "words": 0,
        "phrases": 0,
        "long_sentences": 0,
    }
    char_length_hist: Dict[int, int] = {}
    word_count_hist: Dict[int, int] = {}

    for record in samples:
        category = _classify_text_shape(record)
        category_counts[category] = category_counts.get(category, 0) + 1

        text = record.label_text
        char_len = len(text.replace(" ", "").replace("\n", ""))
        word_len = len([w for w in text.split() if w])

        char_length_hist[char_len] = char_length_hist.get(char_len, 0) + 1
        word_count_hist[word_len] = word_count_hist.get(word_len, 0) + 1

    return {
        "total_samples": len(samples),
        "unique_labels": len({record.label_text for record in samples}),
        "category_counts": category_counts,
        "char_length_histogram": dict(sorted(char_length_hist.items())),
        "word_count_histogram": dict(sorted(word_count_hist.items())),
    }


def build_character_set(samples: Sequence[SampleRecord]) -> List[str]:
    chars = set()
    for record in samples:
        for ch in record.label_text:
            if ch: chars.add(ch)
    return sorted(chars)


class BrahmiDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor=None,
        max_label_length: int = 64,
        map_filename: str = "map.json",
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        image_size: int = 384,
        max_fixed_per_class: int = 100,
        max_words: int = 10000,
        max_phrases: int = 10000,
        max_long_sentences: int = 8000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.processor = processor
        self.max_label_length = max_label_length
        self.image_size = image_size
        self.split = split

        all_samples = load_capped_samples(
            data_dir, 
            map_filename=map_filename,
            max_fixed_per_class=max_fixed_per_class,
            max_words=max_words,
            max_phrases=max_phrases,
            max_long_sentences=max_long_sentences,
            seed=seed
        )
        if not all_samples:
            raise ValueError(f"No image samples found from map.json under: {data_dir}")

        self.all_samples = all_samples
        self.samples = split_samples(all_samples, split=split, split_ratios=split_ratios, seed=seed)
        self.summary = summarize_samples(self.samples)
        self.character_set = build_character_set(all_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))

        image = letterbox_pil(image, target_size=(self.image_size, self.image_size))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        labels = self.processor.tokenizer(
            sample.label_text,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        category_id = ["characters_ngrams", "words", "phrases", "long_sentences"].index(
            _classify_text_shape(sample)
        )
        
        return {
            "pixel_values": pixel_values, 
            "labels": labels,
            "category_id": category_id,
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        import torch
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        # Include category id for evaluation metrics logging
        category_ids = torch.tensor([item["category_id"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels, "category_ids": category_ids}


def _iter_dataset_samples(dataset) -> List[SampleRecord]:
    if isinstance(dataset, BrahmiDataset):
        return list(dataset.samples)

    if isinstance(dataset, ConcatDataset):
        merged: List[SampleRecord] = []
        for subdataset in dataset.datasets:
            if not hasattr(subdataset, "samples"):
                raise TypeError("Weighted sampling requires datasets with a .samples attribute")
            merged.extend(subdataset.samples)
        return merged

    raise TypeError("Unsupported dataset type for weighted sampling")


def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Build a category-balanced sampler for BrahmiDataset or ConcatDataset."""
    samples = _iter_dataset_samples(dataset)
    counts = {
        "characters_ngrams": 0,
        "words": 0,
        "phrases": 0,
        "long_sentences": 0,
    }

    for sample in samples:
        counts[_classify_text_shape(sample)] += 1

    total = max(len(samples), 1)
    cat_weights = {
        cat: (total / count) if count > 0 else 0.0
        for cat, count in counts.items()
    }

    sample_weights = [cat_weights[_classify_text_shape(sample)] for sample in samples]

    import torch
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
