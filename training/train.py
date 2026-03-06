"""
training/train.py — TrOCR Fine-Tuning Script for Brahmi OCR
=============================================================

Fine-tune  microsoft/trocr-small-printed  on the synthetic Brahmi dataset
produced by  dataset/generate_synthetic.py .

USAGE:
    python training/train.py                     # uses defaults
    python training/train.py --epochs 5 --lr 3e-5 --batch_size 4

DEFAULTS:
    model   : microsoft/trocr-small-printed
    epochs  : 10
    lr      : 5e-5
    batch    : 2
    output  : model/brahmi_trocr/
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_loader import BrahmiDataset


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TrOCR on Brahmi OCR data")
    p.add_argument("--data_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"),
                    help="Path to dataset/ directory")
    p.add_argument("--model_name", type=str,
                    default="microsoft/trocr-small-printed",
                    help="HuggingFace model identifier")
    p.add_argument("--output_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "brahmi_trocr"),
                    help="Directory to save the fine-tuned model")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_label_length", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------

def load_model(model_name: str, processor):
    """
    Load TrOCR VisionEncoderDecoderModel and configure its generation params.
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configure special tokens for generation — on BOTH config and generation_config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Generation settings — set on generation_config (NOT model.config)
    # so they are properly saved and loaded during inference
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = 64
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    return model


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    """
    Run one epoch of training.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a validation dataloader.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    print(f"Device : {device}  |  FP16 : {use_fp16}")

    # ---- Processor & model ----
    print(f"Loading processor & model: {args.model_name}")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = load_model(args.model_name, processor)
    model.to(device)

    # ---- Datasets ----
    print(f"Loading datasets from: {args.data_dir}")
    train_ds = BrahmiDataset(args.data_dir, split="train",
                             processor=processor,
                             max_label_length=args.max_label_length)
    val_ds = BrahmiDataset(args.data_dir, split="val",
                           processor=processor,
                           max_label_length=args.max_label_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=BrahmiDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=BrahmiDataset.collate_fn)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    # ---- Training loop ----
    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs}  |  "
              f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  ➜ Best model saved to {args.output_dir}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
