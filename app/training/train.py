#!/usr/bin/env python3
"""
Training script for lip-sync detection model.

Usage:
    python -m app.training.train --data-dir data/AVLips1\ 2 --epochs 50 --batch-size 8
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.device import get_device
from ..core.logger import get_logger
from ..models.lip_sync_model import LipSyncModel
from .dataset import LipSyncDataset
from .collate import safe_collate

logger = get_logger(__name__)


def train_epoch(
    model: LipSyncModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_batches = 0
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=not verbose)
    skipped_batches = 0
    for batch_idx, batch_data in enumerate(pbar):
        # Handle batches where all samples failed (corrupt videos)
        if batch_data is None:
            skipped_batches += 1
            if verbose and skipped_batches == 1:
                logger.warning("⚠️  Skipping batch with all failed samples (corrupt videos). This is normal if your dataset has some bad files.")
            continue
        
        visual, audio, labels = batch_data
        visual = visual.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass (logits)
        logits = model(visual, audio)  # (B,)

        # Loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1

        # Accuracy
        probs = torch.sigmoid(logits)
        pred_binary = (probs > 0.5).float()
        batch_correct = (pred_binary == labels).sum().item()
        batch_total = labels.size(0)
        correct += batch_correct
        total_samples += batch_total

        # Running averages (exponential moving average)
        if num_batches == 1:
            running_loss = batch_loss
            running_acc = batch_correct / batch_total if batch_total > 0 else 0.0
        else:
            running_loss = 0.9 * running_loss + 0.1 * batch_loss
            running_acc = 0.9 * running_acc + 0.1 * (batch_correct / batch_total if batch_total > 0 else 0.0)

        # Update progress bar with detailed metrics
        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg": f"{running_loss:.4f}",
                "acc": f"{running_acc:.2%}",
                "lr": f"{current_lr:.2e}",
            })

        # Log every 50 batches for detailed tracking
        if verbose and (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {batch_loss:.4f} (avg: {running_loss:.4f}) | "
                f"Acc: {batch_correct}/{batch_total} ({batch_correct/batch_total:.2%}) | "
                f"LR: {current_lr:.2e}"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    if skipped_batches > 0 and verbose:
        logger.info(f"⚠️  Skipped {skipped_batches} batches due to corrupt videos (this is normal if your dataset has some bad files)")
    
    return avg_loss, accuracy


def validate(
    model: LipSyncModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Validate and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", disable=not verbose)
        skipped_batches = 0
        for batch_data in pbar:
            # Handle batches where all samples failed (corrupt videos)
            if batch_data is None:
                skipped_batches += 1
                continue
            
            visual, audio, labels = batch_data
            visual = visual.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            logits = model(visual, audio)  # (B,)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Accuracy: threshold at 0.5
            probs = torch.sigmoid(logits)
            pred_binary = (probs > 0.5).float()
            batch_correct = (pred_binary == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total

            # Per-class accuracy (REAL vs FAKE)
            real_mask = labels == 1.0
            fake_mask = labels == 0.0
            if real_mask.any():
                real_correct += (pred_binary[real_mask] == labels[real_mask]).sum().item()
                real_total += real_mask.sum().item()
            if fake_mask.any():
                fake_correct += (pred_binary[fake_mask] == labels[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()

            # Update progress bar
            if verbose:
                running_acc = correct / total if total > 0 else 0.0
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{running_acc:.2%}",
                })

    # Calculate average loss only over processed batches (not skipped ones)
    num_processed_batches = len(dataloader) - skipped_batches
    avg_loss = total_loss / num_processed_batches if num_processed_batches > 0 else 0.0
    
    if skipped_batches > 0 and verbose:
        logger.warning(f"⚠️  Skipped {skipped_batches} validation batches due to corrupt videos")
    accuracy = correct / total if total > 0 else 0.0
    
    if verbose and real_total > 0 and fake_total > 0:
        real_acc = real_correct / real_total if real_total > 0 else 0.0
        fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0
        logger.info(
            f"  Validation details: REAL Acc={real_acc:.2%} ({real_correct}/{real_total}), "
            f"FAKE Acc={fake_acc:.2%} ({fake_correct}/{fake_total})"
        )
    
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lip-sync detection model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing 0_real/ and 1_fake/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show verbose training output")
    parser.add_argument(
        "--no-face-detection",
        action="store_true",
        help="Disable face detection (uses center crop - NOT recommended for production)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience based on accuracy (stop if accuracy doesn't improve for N epochs). If None, training continues for all epochs.",
    )
    args = parser.parse_args()

    # Device
    device = get_device(args.device)
    logger.info("=" * 80)
    logger.info(f"🚀 Training Configuration:")
    logger.info(f"  Device: {device}")
    if device.type == "mps":
        logger.info(f"  ✅ Using Apple Silicon GPU (MPS)")
    elif device.type == "cuda":
        logger.info(f"  ✅ Using NVIDIA GPU (CUDA)")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"  ⚠️  Using CPU (slower - consider using GPU)")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info("=" * 80)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    full_dataset = LipSyncDataset(
        args.data_dir, require_face_detection=not args.no_face_detection
    )
    if args.no_face_detection:
        logger.warning(
            "⚠️  Face detection disabled - using center crop. This is NOT recommended for production training!"
        )
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 2-4 if you have multiple cores
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
    )

    # Model
    model = LipSyncModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {param_count:,} parameters")
    logger.info(f"Model moved to device: {next(model.parameters()).device}")

    # Loss and optimizer (logits)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    if args.resume and args.resume.is_file():
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        
        # Try to load scheduler state if available (for ReduceLROnPlateau, this tracks internal state)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Loaded scheduler state from checkpoint")
        
        # Check if best_model_accuracy.pth exists and has better accuracy than resume checkpoint
        best_acc_path = args.output_dir / "best_model_accuracy.pth"
        if best_acc_path.exists():
            try:
                best_acc_checkpoint = torch.load(best_acc_path, map_location=device)
                best_acc_from_file = best_acc_checkpoint.get("best_val_acc", best_acc_checkpoint.get("val_acc", 0.0))
                if best_acc_from_file > best_val_acc:
                    best_val_acc = best_acc_from_file
                    logger.info(f"Found better accuracy in best_model_accuracy.pth: {best_val_acc:.2%}")
            except Exception as e:
                logger.warning(f"Could not load best_model_accuracy.pth: {e}")
        
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
        logger.info(f"Best validation accuracy so far: {best_val_acc:.2%}")
        logger.info(f"Epochs without improvement: {epochs_without_improvement}")

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose=args.verbose
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, verbose=args.verbose)

        # Detailed epoch summary
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 80)
        logger.info(
            f"Epoch {epoch}/{args.epochs - 1} Summary:"
        )
        logger.info(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%} "
            f"| Val: Loss={val_loss:.4f}, Acc={val_acc:.2%} "
            f"| LR={current_lr:.2e}"
        )
        logger.info("=" * 80)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint (include T so inference knows expected input shape)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "epochs_without_improvement": epochs_without_improvement,
            "video_frames": full_dataset.video_frames,
            "audio_frames": full_dataset.audio_frames,
        }

        # Save latest
        torch.save(checkpoint, args.output_dir / "latest.pth")

        # Save best based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, args.output_dir / "best_model_loss.pth")
            logger.info(f"✅ Saved best model (val_loss={val_loss:.4f})")

        # Save best based on validation accuracy (most accurate model)
        accuracy_improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            accuracy_improved = True
            torch.save(checkpoint, args.output_dir / "best_model_accuracy.pth")
            logger.info(f"🎯 Saved most accurate model (val_acc={val_acc:.2%}, epoch={epoch})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 0:
                logger.info(f"📉 Accuracy not improved for {epochs_without_improvement} epoch(s) (best: {best_val_acc:.2%})")

        # Early stopping based on accuracy degradation
        if args.early_stopping_patience is not None:
            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info("=" * 80)
                logger.info(f"🛑 Early stopping triggered!")
                logger.info(f"   Accuracy hasn't improved for {epochs_without_improvement} epochs")
                logger.info(f"   Best accuracy achieved: {best_val_acc:.2%} at epoch {epoch - epochs_without_improvement}")
                logger.info(f"   Most accurate model saved to: {args.output_dir / 'best_model_accuracy.pth'}")
                logger.info("=" * 80)
                break

    logger.info("Training complete!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"   Best loss model: {args.output_dir / 'best_model_loss.pth'}")
    logger.info(f"   Best accuracy model: {args.output_dir / 'best_model_accuracy.pth'}")


if __name__ == "__main__":
    main()
