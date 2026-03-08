"""
training/dataset_loader.py — PyTorch Dataset for Brahmi OCR
============================================================

Loads Brahmi script images and their Unicode text labels from:
  1. Our synthetic dataset (labels.txt + splits)
  2. Folder-labelled datasets (Capstone / BrahmiGAN) via phonetic mapping

USAGE:
    from dataset_loader import BrahmiDataset, FolderLabeledDataset
    ds1 = BrahmiDataset("dataset", split="train", processor=processor)
    ds2 = FolderLabeledDataset("Capstone_Brahmi_Inscriptions/OCR/OCR_Dataset/RecognizerDataset_150_210", processor=processor)
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class BrahmiDataset(Dataset):
    """
    PyTorch Dataset for loading Brahmi script images from our synthetic dataset.
    Labels come from labels.txt, split by train/val/test.txt.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor=None,
        max_label_length: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.processor = processor
        self.max_label_length = max_label_length

        # ---- Read labels.txt → {filename: text} ----
        labels_path = os.path.join(data_dir, "labels.txt")
        self.label_map = {}
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" in line:
                    fname, text = line.split("\t", maxsplit=1)
                    self.label_map[fname] = text

        # ---- Read split file → list of filenames ----
        split_path = os.path.join(data_dir, "splits", f"{split}.txt")
        with open(split_path, "r", encoding="utf-8") as f:
            split_names = {line.strip() for line in f if line.strip()}

        # ---- Build samples list (keep only filenames in this split) ----
        self.samples = []
        images_dir = os.path.join(data_dir, "images")
        for fname in sorted(split_names):
            if fname in self.label_map:
                img_path = os.path.join(images_dir, fname)
                self.samples.append((img_path, self.label_map[fname]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    @staticmethod
    def collate_fn(batch: list) -> dict:
        import torch
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}


class FolderLabeledDataset(Dataset):
    """
    PyTorch Dataset for folder-labelled Brahmi character datasets.

    Directory structure:
        root_dir/
            ka/       ← folder name is the phonetic label
                img1.tif
                img2.tif
            kaa/
                ...

    Each folder name is converted to Brahmi Unicode via phonetic_mapping.py.
    """

    def __init__(
        self,
        root_dir: str,
        processor=None,
        max_label_length: int = 64,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            root_dir:         Path to the folder-labeled dataset root.
            processor:        A TrOCRProcessor instance.
            max_label_length: Max token length for label encoding.
            split:            'train' or 'val'. We do a random split.
            val_ratio:        Fraction of data to use for validation.
            seed:             Random seed for reproducible splits.
        """
        super().__init__()
        self.processor = processor
        self.max_label_length = max_label_length

        # Import phonetic mappings
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)
        from dataset.phonetic_mapping import PHONETIC_TO_BRAHMI
        from dataset.brahmitGanDatasetMapping import _BRAHMIGAN_MAPPING as PHONETIC_TO_BRAHMI2

        # Check which dataset this is based on path keywords
        root_lower = root_dir.lower()
        is_brahmigan = "brahmigan" in root_lower

        # ---- Collect all (image_path, unicode_label) pairs ----
        all_samples = []
        if not os.path.isdir(root_dir):
            print(f"  ⚠ FolderLabeledDataset: {root_dir} not found, skipping")
            self.samples = []
            return

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Map folder name to Brahmi Unicode using appropriate mapping
            if is_brahmigan:
                brahmi_label = PHONETIC_TO_BRAHMI2.get(folder_name)
            else:
                brahmi_label = PHONETIC_TO_BRAHMI.get(folder_name)
            if brahmi_label is None:
                print(f"  ⚠ Skipping unmapped folder: {folder_name}")
                continue

            # Collect all image files in this folder
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path):
                    all_samples.append((img_path, brahmi_label))

        # ---- Split into train/val ----
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        n_val = int(len(all_samples) * val_ratio)

        if split == "val":
            self.samples = all_samples[:n_val]
        else:  # train
            self.samples = all_samples[n_val:]

        print(f"  FolderLabeledDataset ({split}): {len(self.samples)} samples "
              f"from {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, text = self.samples[idx]

        # Load and convert to RGB (handles TIF, BMP, JPG, PNG)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # If an image fails to load, return a blank white image
            print(f"  ⚠ Failed to load {img_path}: {e}")
            image = Image.new("RGB", (384, 384), (255, 255, 255))

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    @staticmethod
    def collate_fn(batch: list) -> dict:
        import torch
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}


def build_combined_dataset(
    synthetic_dir: str,
    extra_dirs: list,
    processor,
    split: str = "train",
    max_label_length: int = 64,
) -> Dataset:
    """
    Build a combined dataset from synthetic + folder-labelled sources.

    Args:
        synthetic_dir: Path to our synthetic dataset (with labels.txt)
        extra_dirs:    List of paths to folder-labelled datasets
        processor:     TrOCRProcessor
        split:         'train' or 'val'
        max_label_length: Max token length

    Returns:
        A ConcatDataset combining all sources.
    """
    datasets = []

    # 1. Our synthetic dataset
    if os.path.isdir(os.path.join(synthetic_dir, "splits")):
        ds = BrahmiDataset(synthetic_dir, split=split, processor=processor,
                           max_label_length=max_label_length)
        datasets.append(ds)
        print(f"  Synthetic ({split}): {len(ds)} samples")

    # 2. Extra folder-labelled datasets
    for extra_dir in extra_dirs:
        if os.path.isdir(extra_dir):
            ds = FolderLabeledDataset(
                extra_dir, processor=processor,
                max_label_length=max_label_length, split=split,
            )
            if len(ds) > 0:
                datasets.append(ds)

    if len(datasets) == 0:
        raise ValueError("No datasets found!")

    combined = ConcatDataset(datasets)
    print(f"  Combined {split} dataset: {len(combined)} total samples")
    return combined
