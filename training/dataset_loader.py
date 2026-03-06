"""
training/dataset_loader.py — PyTorch Dataset for Brahmi OCR
============================================================

Loads Brahmi script images and their Unicode text labels from the
dataset produced by  dataset/generate_synthetic.py .

DATA FORMAT:
    - Images    : dataset/images/*.png
    - Labels    : dataset/labels.txt  (TAB-separated: filename<TAB>text)
    - Splits    : dataset/splits/{train,val,test}.txt  (one filename per line)

USAGE:
    from dataset_loader import BrahmiDataset
    ds = BrahmiDataset("dataset", split="train", processor=processor)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=ds.collate_fn)
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class BrahmiDataset(Dataset):
    """
    PyTorch Dataset for loading Brahmi script images and labels.

    Each sample is returned as a dict::

        {
            "pixel_values": Tensor  [C, H, W],
            "labels":       Tensor  [seq_len],
        }
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor=None,
        max_label_length: int = 64,
    ):
        """
        Args:
            data_dir:         Root dataset/ directory.
            split:            'train', 'val', or 'test'.
            processor:        A TrOCRProcessor instance.
            max_label_length: Max token length for label encoding.
        """
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

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        # Encode image through the TrOCR processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenise the label text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so CrossEntropyLoss ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Custom collate that stacks pixel_values and labels into batches.
        """
        import torch

        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}
