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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..core.device import get_device
from ..core.logger import get_logger
from ..models.lip_sync_model import LipSyncModel
from .dataset import LipSyncDataset
from .augmentation import AugmentedLipSyncDataset
from .collate import safe_collate

logger = get_logger(__name__)


def find_best_threshold(probs: list[float], labels: list[float]) -> tuple[float, float]:
    """Sweep thresholds 0.05 → 0.95 and return threshold that maximizes F1."""
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0.05, 0.95, 91)
    probs_np = np.array(probs, dtype=np.float64)
    labels_np = np.array(labels, dtype=np.float64).astype(int)

    for t in thresholds:
        preds = (probs_np > t).astype(int)
        tp = ((preds == 1) & (labels_np == 1)).sum()
        fp = ((preds == 1) & (labels_np == 0)).sum()
        fn = ((preds == 0) & (labels_np == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, best_f1


def _get_samples_from_dataset(dataset: Any) -> list | None:
    """Get (path, label) samples from full dataset or wrapped dataset."""
    if hasattr(dataset, "samples"):
        return dataset.samples
    if hasattr(dataset, "base_dataset"):
        return getattr(dataset.base_dataset, "samples", None)
    return None


def log_class_distribution(
    full_dataset: Any,
    train_dataset: Subset,
    val_dataset: Subset,
) -> None:
    """Log REAL/FAKE counts for train and validation before training."""
    samples = _get_samples_from_dataset(full_dataset)
    if not samples:
        logger.warning("Could not infer class distribution (no samples list on dataset)")
        return
    train_real = sum(1 for i in train_dataset.indices if samples[i][1] == 1)
    train_fake = len(train_dataset) - train_real
    val_real = sum(1 for i in val_dataset.indices if samples[i][1] == 1)
    val_fake = len(val_dataset) - val_real
    logger.info("=" * 80)
    logger.info("📊 Class distribution (before training):")
    logger.info(f"  Train: REAL={train_real}, FAKE={train_fake} (total={len(train_dataset)})")
    logger.info(f"  Val:   REAL={val_real}, FAKE={val_fake} (total={len(val_dataset)})")
    logger.info("=" * 80)


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
) -> tuple[float, float, dict[str, int | float]]:
    """Validate and return loss, accuracy, and metrics (TP, TN, FP, FN, F1, best_threshold, etc.)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = tn = fp = fn = 0
    all_probs: list[float] = []
    all_labels: list[float] = []

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
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

            pred_binary = (probs > 0.5).float()
            batch_correct = (pred_binary == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total

            # Confusion matrix counts (label 1 = REAL, 0 = FAKE)
            real_mask = labels == 1.0
            fake_mask = labels == 0.0
            pred_real = pred_binary == 1.0
            pred_fake = pred_binary == 0.0
            tp += ((pred_real) & (real_mask)).sum().item()
            tn += ((pred_fake) & (fake_mask)).sum().item()
            fp += ((pred_real) & (fake_mask)).sum().item()
            fn += ((pred_fake) & (real_mask)).sum().item()

            if verbose:
                running_acc = correct / total if total > 0 else 0.0
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{running_acc:.2%}",
                })

    # Average loss
    num_processed_batches = len(dataloader) - skipped_batches
    avg_loss = total_loss / num_processed_batches if num_processed_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # F1, precision, recall at fixed threshold 0.5
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Threshold sweep: find threshold that maximizes F1 on validation
    best_threshold = 0.5
    best_f1_at_threshold = f1
    if all_probs and all_labels:
        best_threshold, best_f1_at_threshold = find_best_threshold(all_probs, all_labels)
        if verbose:
            logger.info(
                f"  🎯 Best threshold (val) = {best_threshold:.3f} | F1 at threshold = {best_f1_at_threshold:.2%}"
            )

    metrics = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_threshold": best_threshold,
        "best_f1_at_threshold": best_f1_at_threshold,
    }

    if skipped_batches > 0 and verbose:
        logger.warning(f"⚠️  Skipped {skipped_batches} validation batches due to corrupt videos")

    if verbose:
        logger.info(
            f"  Validation (threshold=0.5): TP={tp}, TN={tn}, FP={fp}, FN={fn} | "
            f"Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}"
        )

    return avg_loss, accuracy, metrics


def save_confusion_matrix_epoch(
    output_dir: Path,
    epoch: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> None:
    """Save confusion matrix for an epoch to a text file. Rows=true, cols=pred; (fake, real)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_epoch_{epoch:03d}.txt"
    # Rows = true class (0=fake, 1=real), Cols = predicted (0=fake, 1=real)
    #         pred_fake  pred_real
    # true_fake   TN        FP
    # true_real   FN        TP
    lines = [
        "# Confusion matrix (rows=true, cols=pred); REAL=1, FAKE=0",
        "#             pred_fake  pred_real",
        f"# true_fake   {tn:>8}  {fp:>8}",
        f"# true_real   {fn:>8}  {tp:>8}",
        "",
        f"TN={tn}  FP={fp}",
        f"FN={fn}  TP={tp}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  Confusion matrix saved: {path}")


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
    log_class_distribution(full_dataset, train_dataset, val_dataset)

    confusion_matrices_dir = args.output_dir / "confusion_matrices"
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

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
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    # Phase 1: Train classifier with frozen encoders
    logger.info("Phase 1: Training classifier (encoders frozen)")
    freeze_encoder(model, freeze_visual=True, freeze_audio=True)

    for epoch in range(args.freeze_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose=args.verbose
        )
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device, verbose=args.verbose)

        # Log FP/FN and save confusion matrix this epoch
        logger.info(f"  Epoch {epoch} Val: FP={val_metrics['fp']}, FN={val_metrics['fn']}")
        save_confusion_matrix_epoch(
            confusion_matrices_dir,
            epoch,
            val_metrics["tp"],
            val_metrics["tn"],
            val_metrics["fp"],
            val_metrics["fn"],
        )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 80)
        logger.info(
            f"Phase 1 - Epoch {epoch}/{args.freeze_epochs - 1} (Encoders Frozen):"
        )
        logger.info(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%}, F1@0.5={val_metrics['f1']:.2%}, F1@best={val_metrics['best_f1_at_threshold']:.2%} (t={val_metrics['best_threshold']:.2f}) | LR={current_lr:.2e}"
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

        # Save best based on validation F1 (using threshold-tuned F1)
        if val_metrics["best_f1_at_threshold"] > best_val_f1:
            best_val_f1 = val_metrics["best_f1_at_threshold"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_metrics["best_f1_at_threshold"],
                    "best_val_f1": best_val_f1,
                    "best_threshold": val_metrics["best_threshold"],
                },
                args.output_dir / "best_frozen_f1.pth",
            )
            logger.info(
                f"🎯 Saved best F1 frozen model (val_f1={best_val_f1:.2%}, threshold={val_metrics['best_threshold']:.3f}, epoch={epoch})"
            )

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
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device, verbose=args.verbose)

        # Log FP/FN and save confusion matrix this epoch
        logger.info(f"  Epoch {epoch} Val: FP={val_metrics['fp']}, FN={val_metrics['fn']}")
        save_confusion_matrix_epoch(
            confusion_matrices_dir,
            epoch,
            val_metrics["tp"],
            val_metrics["tn"],
            val_metrics["fp"],
            val_metrics["fn"],
        )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 80)
        logger.info(
            f"Phase 2 - Epoch {epoch}/{args.epochs - 1} (Encoders Unfrozen):"
        )
        logger.info(
            f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
            f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%}, F1@0.5={val_metrics['f1']:.2%}, F1@best={val_metrics['best_f1_at_threshold']:.2%} (t={val_metrics['best_threshold']:.2f}) | LR={current_lr:.2e}"
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

        # Save best based on validation F1 (using threshold-tuned F1)
        if val_metrics["best_f1_at_threshold"] > best_val_f1:
            best_val_f1 = val_metrics["best_f1_at_threshold"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_metrics["best_f1_at_threshold"],
                    "best_val_f1": best_val_f1,
                    "best_threshold": val_metrics["best_threshold"],
                },
                args.output_dir / "best_model_f1.pth",
            )
            logger.info(
                f"🎯 Saved best F1 model (val_f1={best_val_f1:.2%}, threshold={val_metrics['best_threshold']:.3f}, epoch={epoch})"
            )

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
    logger.info(f"   Best validation F1: {best_val_f1:.2%}")
    logger.info(f"   Best loss model: {args.output_dir / 'best_model_loss.pth'}")
    logger.info(f"   Best accuracy model: {args.output_dir / 'best_model_accuracy.pth'}")
    logger.info(f"   Best F1 model: {args.output_dir / 'best_model_f1.pth'}")
    logger.info(f"   Confusion matrices per epoch: {confusion_matrices_dir}")


if __name__ == "__main__":
    main()
