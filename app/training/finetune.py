#!/usr/bin/env python3
"""
Fine-tuning script for production-grade lip-sync manipulation detection.

Supports:
- Transfer learning from pre-trained weights
- Progressive unfreezing
- Data augmentation for robustness
- Multi-angle and multi-face training
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.device import get_device
from ..core.logger import get_logger
from ..models.lip_sync_model import LipSyncModel
from .dataset import LipSyncDataset
from .augmentation import AugmentedLipSyncDataset
from .collate import safe_collate

logger = get_logger(__name__)


def freeze_encoder(model: LipSyncModel, freeze_visual: bool = True, freeze_audio: bool = True) -> None:
    """Freeze encoder weights for transfer learning."""
    if freeze_visual:
        for param in model.visual_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen visual encoder")

    if freeze_audio:
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen audio encoder")


def unfreeze_encoder(model: LipSyncModel, unfreeze_visual: bool = True, unfreeze_audio: bool = True) -> None:
    """Unfreeze encoder weights for fine-tuning."""
    if unfreeze_visual:
        for param in model.visual_encoder.parameters():
            param.requires_grad = True
        logger.info("Unfrozen visual encoder")

    if unfreeze_audio:
        for param in model.audio_encoder.parameters():
            param.requires_grad = True
        logger.info("Unfrozen audio encoder")


def train_epoch(
    model: LipSyncModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose: bool = True,
) -> tuple[float, float]:
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
        logits = model(visual, audio)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Running averages
        if num_batches == 1:
            running_loss = batch_loss
            running_acc = batch_correct / batch_total if batch_total > 0 else 0.0
        else:
            running_loss = 0.9 * running_loss + 0.1 * batch_loss
            running_acc = 0.9 * running_acc + 0.1 * (batch_correct / batch_total if batch_total > 0 else 0.0)

        # Update progress bar
        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg": f"{running_loss:.4f}",
                "acc": f"{running_acc:.2%}",
                "lr": f"{current_lr:.2e}",
                "grad": f"{grad_norm:.2f}",
            })

        # Log every 50 batches
        if verbose and (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {batch_loss:.4f} (avg: {running_loss:.4f}) | "
                f"Acc: {batch_correct}/{batch_total} ({batch_correct/batch_total:.2%}) | "
                f"LR: {current_lr:.2e} | Grad: {grad_norm:.2f}"
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
) -> tuple[float, float]:
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

            logits = model(visual, audio)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            pred_binary = (probs > 0.5).float()
            batch_correct = (pred_binary == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total

            # Per-class accuracy
            real_mask = labels == 1.0
            fake_mask = labels == 0.0
            if real_mask.any():
                real_correct += (pred_binary[real_mask] == labels[real_mask]).sum().item()
                real_total += real_mask.sum().item()
            if fake_mask.any():
                fake_correct += (pred_binary[fake_mask] == labels[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()

            if verbose:
                running_acc = correct / total if total > 0 else 0.0
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{running_acc:.2%}",
                })

    # Calculate average loss only over processed batches (not skipped ones)
    num_processed_batches = len(dataloader) - skipped_batches
    avg_loss = total_loss / num_processed_batches if num_processed_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    if skipped_batches > 0 and verbose:
        logger.warning(f"⚠️  Skipped {skipped_batches} validation batches due to corrupt videos")
    
    if verbose and real_total > 0 and fake_total > 0:
        real_acc = real_correct / real_total if real_total > 0 else 0.0
        fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0
        logger.info(
            f"  Validation details: REAL Acc={real_acc:.2%} ({real_correct}/{real_total}), "
            f"FAKE Acc={fake_acc:.2%} ({fake_correct}/{fake_total})"
        )
    
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune lip-sync manipulation detection model")
    parser.add_argument("--data-dir", type=Path, required=True, help="Training data directory")
    parser.add_argument("--pretrained", type=Path, help="Path to pre-trained weights")
    parser.add_argument("--output-dir", type=Path, default=Path("weights"), help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs")
    parser.add_argument("--freeze-epochs", type=int, default=10, help="Epochs with frozen encoders")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr-encoder", type=float, default=1e-5, help="LR for encoders when unfrozen")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show verbose training output")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience based on accuracy (stop if accuracy doesn't improve for N epochs). If None, training continues for all epochs.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    logger.info("=" * 80)
    logger.info(f"🚀 Fine-Tuning Configuration:")
    logger.info(f"  Device: {device}")
    if device.type == "mps":
        logger.info(f"  ✅ Using Apple Silicon GPU (MPS)")
    elif device.type == "cuda":
        logger.info(f"  ✅ Using NVIDIA GPU (CUDA)")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"  ⚠️  Using CPU (slower - consider using GPU)")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr} (encoders: {args.lr_encoder})")
    logger.info("=" * 80)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset with optional augmentation
    if args.use_augmentation:
        full_dataset = AugmentedLipSyncDataset(args.data_dir)
        logger.info("Using data augmentation")
    else:
        full_dataset = LipSyncDataset(args.data_dir)

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    model = LipSyncModel(detect_artifacts=True).to(device)

    # Load pre-trained weights if provided
    if args.pretrained and args.pretrained.is_file():
        logger.info(f"Loading pre-trained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Pre-trained weights loaded")

    # Loss and optimizer (logits)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": args.lr},
            {"params": model.fusion.parameters(), "lr": args.lr},
            {"params": model.projection.parameters(), "lr": args.lr},
            {"params": model.temporal.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    if model.artifact_detector:
        optimizer.add_param_group({"params": model.artifact_detector.parameters(), "lr": args.lr})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Phase 1: Train classifier with frozen encoders
    logger.info("Phase 1: Training classifier (encoders frozen)")
    freeze_encoder(model, freeze_visual=True, freeze_audio=True)

    for epoch in range(args.freeze_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose=args.verbose
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, verbose=args.verbose)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 80)
        logger.info(
            f"Phase 1 - Epoch {epoch}/{args.freeze_epochs - 1} (Encoders Frozen):"
        )
        logger.info(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%} | LR={current_lr:.2e}"
        )
        logger.info("=" * 80)
        scheduler.step(val_loss)

        # Save best based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                args.output_dir / "best_frozen_loss.pth",
            )

        # Save best based on validation accuracy (most accurate model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                args.output_dir / "best_frozen_accuracy.pth",
            )
            logger.info(f"🎯 Saved most accurate frozen model (val_acc={val_acc:.2%}, epoch={epoch})")
        else:
            epochs_without_improvement += 1

    # Phase 2: Fine-tune encoders
    logger.info("Phase 2: Fine-tuning encoders (unfrozen)")
    unfreeze_encoder(model, unfreeze_visual=True, unfreeze_audio=True)

    # Update optimizer with encoder parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": model.visual_encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.audio_encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.classifier.parameters(), "lr": args.lr},
            {"params": model.fusion.parameters(), "lr": args.lr},
            {"params": model.projection.parameters(), "lr": args.lr},
            {"params": model.temporal.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    if model.artifact_detector:
        optimizer.add_param_group({"params": model.artifact_detector.parameters(), "lr": args.lr})

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Reset accuracy tracking for Phase 2 (or continue tracking)
    phase2_best_val_acc = best_val_acc
    phase2_epochs_without_improvement = epochs_without_improvement

    for epoch in range(args.freeze_epochs, args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose=args.verbose
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, verbose=args.verbose)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 80)
        logger.info(
            f"Phase 2 - Epoch {epoch}/{args.epochs - 1} (Encoders Unfrozen):"
        )
        logger.info(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%} | LR={current_lr:.2e}"
        )
        logger.info("=" * 80)
        scheduler.step(val_loss)

        # Save best based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                args.output_dir / "best_model_loss.pth",
            )

        # Save best based on validation accuracy (most accurate model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            phase2_epochs_without_improvement = 0
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                args.output_dir / "best_model_accuracy.pth",
            )
            logger.info(f"🎯 Saved most accurate model (val_acc={val_acc:.2%}, epoch={epoch})")
        else:
            phase2_epochs_without_improvement += 1
            epochs_without_improvement += 1
            if phase2_epochs_without_improvement > 0:
                logger.info(f"📉 Accuracy not improved for {phase2_epochs_without_improvement} epoch(s) (best: {best_val_acc:.2%})")

        # Early stopping based on accuracy degradation
        if args.early_stopping_patience is not None:
            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info("=" * 80)
                logger.info(f"🛑 Early stopping triggered!")
                logger.info(f"   Accuracy hasn't improved for {epochs_without_improvement} epochs")
                logger.info(f"   Best accuracy achieved: {best_val_acc:.2%}")
                logger.info(f"   Most accurate model saved to: {args.output_dir / 'best_model_accuracy.pth'}")
                logger.info("=" * 80)
                break

    logger.info("Fine-tuning complete!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"   Best loss model: {args.output_dir / 'best_model_loss.pth'}")
    logger.info(f"   Best accuracy model: {args.output_dir / 'best_model_accuracy.pth'}")


if __name__ == "__main__":
    main()
