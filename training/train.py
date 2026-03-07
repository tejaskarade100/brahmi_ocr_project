"""
training/train.py — TrOCR Fine-Tuning Script for Brahmi OCR
=============================================================

Fine-tune  microsoft/trocr-small-printed  on the synthetic Brahmi dataset
produced by  dataset/generate_synthetic.py , optionally combined with
folder-labelled datasets (Capstone, BrahmiGAN).

USAGE:
    python training/train.py                     # uses defaults (synthetic only)
    python training/train.py --extra_data path/to/RecognizerDataset  # + Capstone
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
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_loader import BrahmiDataset, build_combined_dataset


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
    p.add_argument("--output_dir", type=str, default="model/brahmi_trocr",
                        help="Directory to save the trained model.")
    p.add_argument("--drive_save_path", type=str, default="",
                        help="Optional Google Drive path to auto-save the best model at each epoch.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_label_length", type=int, default=128)
    p.add_argument("--cer_eval_batches", type=int, default=0,
                    help="How many val batches to decode for CER (0 = full val set).")
    p.add_argument("--disable_length_balance", action="store_true",
                    help="Disable length-aware weighted sampling on combined datasets.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra_data", type=str, nargs="*", default=[],
                    help="Paths to extra folder-labelled datasets (Capstone, BrahmiGAN)")
    p.add_argument("--resume", action="store_true",
                    help="Resume training from last checkpoint if available.")
    p.add_argument("--resume_checkpoint", type=str, default="",
                    help="Optional explicit path to a training-state checkpoint (.pt).")
    return p.parse_args()


# --------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------

def get_brahmi_characters() -> list:
    """
    Return all Brahmi Unicode characters used in our dataset.
    Must match the character ranges in dataset/generate_synthetic.py.
    """
    chars = []
    # Independent vowels  U+11005 – U+11012
    chars += [chr(c) for c in range(0x11005, 0x11013)]
    # Consonants  U+11013 – U+11037
    chars += [chr(c) for c in range(0x11013, 0x11038)]
    # Dependent vowel signs  U+11038 – U+11046
    chars += [chr(c) for c in range(0x11038, 0x11047)]
    # Digits  U+11052 – U+11069
    chars += [chr(c) for c in range(0x11052, 0x1106A)]
    # Space (used in phrases)
    chars += [" "]
    return chars


def load_model(model_name: str, processor, max_gen_tokens: int = 128):
    """
    Load TrOCR VisionEncoderDecoderModel, add Brahmi tokens, and configure
    generation params.
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # ---- Add Brahmi characters to tokenizer vocabulary ----
    # The default XLMRoBERTa tokenizer encodes Brahmi as <unk>.
    # We add every Brahmi character as a new token so the model
    # can learn to generate real Brahmi IDs instead of <unk>.
    from transformers import AddedToken

    brahmi_chars = get_brahmi_characters()
    existing_vocab = processor.tokenizer.get_vocab()
    new_tokens = [AddedToken(ch, normalized=False) for ch in brahmi_chars if ch not in existing_vocab]
    if new_tokens:
        num_added = processor.tokenizer.add_tokens(new_tokens)
        print(f"  Added {num_added} Brahmi tokens to tokenizer "
              f"(vocab size → {len(processor.tokenizer)})")

    # Resize decoder embeddings to match new vocab size
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder.vocab_size = len(processor.tokenizer)
    model.config.vocab_size = len(processor.tokenizer)

    # Configure special tokens — on BOTH config and generation_config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Generation settings — on generation_config so they are saved properly
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = max_gen_tokens
    model.generation_config.early_stopping = False
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 1

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


def build_length_balanced_sampler(dataset):
    """
    Build a WeightedRandomSampler that upsamples longer labels and phrases.
    This reduces character-only dominance when training on mixed datasets.
    """
    if not isinstance(dataset, ConcatDataset):
        return None

    sample_weights = []
    for sub_ds in dataset.datasets:
        if not hasattr(sub_ds, "samples"):
            sample_weights.extend([1.0] * len(sub_ds))
            continue

        for _, text in sub_ds.samples:
            seq = str(text)
            clean_len = len(seq.replace(" ", ""))

            if clean_len <= 2:
                weight = 0.35
            elif clean_len <= 5:
                weight = 0.75
            elif clean_len <= 10:
                weight = 1.30
            else:
                weight = 2.00

            if " " in seq:
                weight *= 1.20

            sample_weights.append(weight)

    if len(sample_weights) != len(dataset):
        print("  ⚠ Length-balance sampler skipped (weight/sample mismatch).")
        return None

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_cer(preds, labels):
    import editdistance
    total_edits = 0
    total_chars = 0
    for p, l in zip(preds, labels):
        total_edits += editdistance.eval(p, l)
        total_chars += len(l)
    return total_edits / max(total_chars, 1)

def evaluate(
    model,
    dataloader,
    device,
    processor,
    cer_eval_batches: int = 0,
    max_new_tokens: int = 128,
):
    """
    Evaluate the model on a validation dataloader.
    Returns: Average validation loss, and estimated CER (sampled).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            
            # Decode either full validation set or the requested subset.
            if cer_eval_batches <= 0 or i < cer_eval_batches:
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )
                preds_str = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Decode original labels
                labels_clone = labels.clone()
                labels_clone[labels_clone == -100] = processor.tokenizer.pad_token_id
                labels_str = processor.batch_decode(labels_clone, skip_special_tokens=True)
                
                all_preds.extend(preds_str)
                all_labels.extend(labels_str)
                
            num_batches += 1

    try:
        import editdistance
        cer = compute_cer(all_preds, all_labels)
    except ImportError:
        cer = -1.0  # Just return -1 if editdistance is not installed

    return total_loss / max(num_batches, 1), cer


def resolve_resume_checkpoint_path(args) -> str:
    """
    Return the checkpoint path used for resume/load of training state.
    """
    if args.resume_checkpoint:
        return args.resume_checkpoint
    return os.path.join(args.output_dir, "last_checkpoint.pt")


def maybe_resume_training(model, optimizer, scaler, device, checkpoint_path: str, enabled: bool):
    """
    Optionally resume model + optimizer + scaler states and return
    (start_epoch, best_val_cer, best_val_loss).
    """
    start_epoch = 1
    best_val_cer = float("inf")
    best_val_loss = float("inf")

    if not enabled:
        return start_epoch, best_val_cer, best_val_loss

    if not os.path.isfile(checkpoint_path):
        print(f"Resume requested, but checkpoint not found: {checkpoint_path}")
        print("Starting training from scratch.")
        return start_epoch, best_val_cer, best_val_loss

    print(f"Resuming training from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_cer = float(ckpt.get("best_val_cer", float("inf")))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in ckpt:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

    print(f"  ➜ Resume state loaded (next epoch: {start_epoch})")
    return start_epoch, best_val_cer, best_val_loss


def save_training_state(
    checkpoint_path: str,
    model,
    optimizer,
    scaler,
    epoch: int,
    best_val_cer: float,
    best_val_loss: float,
):
    """
    Save latest full training state for reliable resume after interruptions.
    """
    state = {
        "epoch": epoch,
        "best_val_cer": best_val_cer,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_path)


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
    processor = TrOCRProcessor.from_pretrained(args.model_name, use_fast=False)
    model = load_model(args.model_name, processor, max_gen_tokens=args.max_label_length)
    model.to(device)

    # ---- Datasets ----
    print(f"Loading datasets from: {args.data_dir}")
    if args.extra_data:
        print(f"Extra datasets: {args.extra_data}")
        train_ds = build_combined_dataset(
            args.data_dir, args.extra_data, processor,
            split="train", max_label_length=args.max_label_length)
        val_ds = build_combined_dataset(
            args.data_dir, args.extra_data, processor,
            split="val", max_label_length=args.max_label_length)
    else:
        train_ds = BrahmiDataset(args.data_dir, split="train",
                                 processor=processor,
                                 max_label_length=args.max_label_length)
        val_ds = BrahmiDataset(args.data_dir, split="val",
                               processor=processor,
                               max_label_length=args.max_label_length)

    train_sampler = None
    if not args.disable_length_balance:
        train_sampler = build_length_balanced_sampler(train_ds)
        if train_sampler is not None:
            print("Using length-balanced sampler for training.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=BrahmiDataset.collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=BrahmiDataset.collate_fn)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    os.makedirs(args.output_dir, exist_ok=True)
    last_ckpt_path = resolve_resume_checkpoint_path(args)
    last_ckpt_dir = os.path.dirname(last_ckpt_path)
    if last_ckpt_dir:
        os.makedirs(last_ckpt_dir, exist_ok=True)

    # ---- Training loop ----
    start_epoch, best_val_cer, best_val_loss = maybe_resume_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        checkpoint_path=last_ckpt_path,
        enabled=args.resume,
    )

    if start_epoch > args.epochs:
        print(
            f"Checkpoint already at epoch {start_epoch - 1}, "
            f"which meets/exceeds requested --epochs {args.epochs}. Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_cer = evaluate(
            model, val_loader, device, processor,
            cer_eval_batches=args.cer_eval_batches,
            max_new_tokens=args.max_label_length,
        )

        cer_str = f"{val_cer:.4f}" if val_cer >= 0 else "N/A (pip install editdistance)"
        print(f"Epoch {epoch}/{args.epochs}  |  "
              f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val CER: {cer_str}")

        # Save best model primarily by CER (or val_loss as fallback)
        improved = False
        if val_cer >= 0:
            if (val_cer < best_val_cer) or (val_cer == best_val_cer and val_loss < best_val_loss):
                best_val_cer = val_cer
                best_val_loss = val_loss
                improved = True
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True

        if improved:

            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"  ➜ Best model saved to {args.output_dir}")

            if args.drive_save_path:
                try:
                    import shutil
                    if os.path.exists(args.drive_save_path):
                        shutil.rmtree(args.drive_save_path)
                    shutil.copytree(args.output_dir, args.drive_save_path)
                    print(f"  ➜ Auto-saved backup to {args.drive_save_path}!")
                except Exception as e:
                    print(f"  ➜ Warning: Could not auto-save to Drive: {e}")

        save_training_state(
            checkpoint_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_cer=best_val_cer,
            best_val_loss=best_val_loss,
        )
        print(f"  ➜ Latest checkpoint saved to {last_ckpt_path}")

    cer_summary = f"{best_val_cer:.4f}" if best_val_cer != float("inf") else "N/A"
    print(f"\nTraining complete. Best val CER: {cer_summary} | Best val loss: {best_val_loss:.4f}")
    print(f"Model saved at: {args.output_dir}")
    print(f"Latest resume checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
