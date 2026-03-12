#!/usr/bin/env python3
"""
Training script for lip-sync detection model.

Usage:
    python -m app.training.train --data-dir \"data/AVLips1 2\" --epochs 50 --batch-size 8
"""

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from ..core.device import get_device
from ..core.logger import get_logger
from ..models.lip_sync_model import LipSyncModel
from .augmentation import AugmentedLipSyncDataset
from .collate import safe_collate
from .dataset import LipSyncDataset
from .losses import cross_modal_contrastive_loss, sync_contrastive_loss

logger = get_logger(__name__)


def _shift_audio(audio: torch.Tensor, shift_frames: int) -> torch.Tensor:
    """Shift audio along time (last dim). (B, 1, F, T) -> same shape, rolled by shift_frames."""
    if shift_frames == 0:
        return audio
    return torch.roll(audio, shifts=shift_frames, dims=-1)


class ModeOverrideSubset(Dataset):
    """
    Subset wrapper that can override random/center clip sampling mode.
    """

    def __init__(self, subset: Subset, train_mode: bool) -> None:
        self.subset = subset
        self.train_mode = bool(train_mode)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        base_dataset = self.subset.dataset
        real_idx = self.subset.indices[idx]
        if hasattr(base_dataset, "get_item"):
            return base_dataset.get_item(real_idx, train_mode_override=self.train_mode)
        return base_dataset[real_idx]


def _freeze_encoder(
    model: LipSyncModel, freeze_visual: bool = True, freeze_audio: bool = True
) -> None:
    """Freeze encoder parameters (for phased training)."""
    if freeze_visual:
        for param in model.visual_encoder.parameters():
            param.requires_grad = False
    if freeze_audio:
        for param in model.audio_encoder.parameters():
            param.requires_grad = False


def _unfreeze_encoder(
    model: LipSyncModel, unfreeze_visual: bool = True, unfreeze_audio: bool = True
) -> None:
    """Unfreeze encoder parameters (for phased training)."""
    if unfreeze_visual:
        for param in model.visual_encoder.parameters():
            param.requires_grad = True
    if unfreeze_audio:
        for param in model.audio_encoder.parameters():
            param.requires_grad = True


def _trainable_param_groups(
    model: LipSyncModel,
    phase: int,
    lr_head: float,
    lr_encoder: float,
) -> list[dict]:
    """
    Return optimizer param groups for the given phase.
    Phase 1: fusion + classifier only (encoders frozen).
    Phase 2: + audio encoder.
    Phase 3: full model (visual + audio encoders).
    """
    head_params = [
        model.projection.parameters(),
        model.cross_modal.parameters(),
        model.temporal.parameters(),
        model.classifier.parameters(),
    ]
    if model.artifact_detector is not None:
        head_params.append(model.artifact_detector.parameters())

    if phase == 1:
        return [{"params": [p for params in head_params for p in params], "lr": lr_head}]
    if phase == 2:
        groups = [{"params": [p for params in head_params for p in params], "lr": lr_head}]
        groups.append({"params": model.audio_encoder.parameters(), "lr": lr_encoder})
        return groups
    # phase 3: full model
    groups = [{"params": [p for params in head_params for p in params], "lr": lr_head}]
    groups.append({"params": model.audio_encoder.parameters(), "lr": lr_encoder})
    groups.append({"params": model.visual_encoder.parameters(), "lr": lr_encoder})
    return groups


def train_epoch(
    model: LipSyncModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    contrastive_weight: float = 0.1,
    contrastive_temperature: float = 0.07,
    contrastive_fake_margin: float = 0.10,
    sync_weight: float = 0.0,
    sync_shift_frames: Tuple[int, ...] = (5, 10, 15),
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

        # Forward pass with auxiliary representations for contrastive training.
        logits, aux = model(visual, audio, return_aux=True)  # type: ignore[assignment]

        bce_loss = criterion(logits, labels)
        contrastive_loss = cross_modal_contrastive_loss(
            aux["visual_tokens"],
            aux["audio_tokens"],
            labels,
            temperature=contrastive_temperature,
            fake_margin=contrastive_fake_margin,
        )
        loss = bce_loss + contrastive_weight * contrastive_loss

        # Sync contrastive (alignment): (video, correct_audio) vs (video, shifted_audio); real samples only.
        if sync_weight > 0 and sync_shift_frames and (labels >= 0.5).any():
            shifts = [s for s in sync_shift_frames if s != 0] + [-s for s in sync_shift_frames if s != 0]
            shift = random.choice(shifts) if shifts else 0
            if shift != 0:
                audio_shifted = _shift_audio(audio, shift)
                _, aux_neg = model(visual, audio_shifted, return_aux=True)  # type: ignore[assignment]
                sync_loss = sync_contrastive_loss(
                    aux["visual_tokens"],
                    aux["audio_tokens"],
                    [aux_neg["audio_tokens"]],
                    real_mask=(labels >= 0.5),
                    temperature=contrastive_temperature,
                )
                loss = loss + sync_weight * sync_loss

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
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.1,
        help="Weight for cross-modal contrastive auxiliary loss",
    )
    parser.add_argument(
        "--contrastive-temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive similarity scaling",
    )
    parser.add_argument(
        "--contrastive-fake-margin",
        type=float,
        default=0.10,
        help="Margin for fake-pair contrastive separation",
    )
    parser.add_argument(
        "--sync-weight",
        type=float,
        default=0.2,
        help="Weight for sync contrastive loss (video vs shifted-audio). 0 to disable. Default 0.2.",
    )
    parser.add_argument(
        "--sync-shift-frames",
        type=str,
        default="5,10,15",
        help="Comma-separated frame shifts for sync negatives (e.g. 5,10,15). Default 5,10,15.",
    )
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
        "--use-augmentation",
        action="store_true",
        help="Apply data augmentation (flip, rotation, color jitter, noise, speed) to the training split.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Optional path to precomputed tensors directory (manifest.jsonl).",
    )
    parser.add_argument(
        "--storage-format",
        type=str,
        default="npy",
        choices=["npy", "lmdb", "zarr"],
        help="Storage backend when --preprocessed-dir is provided.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience based on accuracy (stop if accuracy doesn't improve for N epochs). If None, training continues for all epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. Safe to set >0 when using --preprocessed-dir "
            "(no MediaPipe in hot path). Keep 0 when training from raw video (MediaPipe "
            "is not fork-safe)."
        ),
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Disable phased encoder freezing; train full model from epoch 0.",
    )
    parser.add_argument(
        "--freeze-phase1-end",
        type=int,
        default=5,
        help="End epoch of Phase 1 (fusion + classifier only). Epochs 0 to this-1. Default 5.",
    )
    parser.add_argument(
        "--freeze-phase2-end",
        type=int,
        default=15,
        help="End epoch of Phase 2 (+ audio encoder). Phase 3 (full model) from this epoch. Default 15.",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=1e-5,
        help="Learning rate for encoders when unfrozen in Phase 2/3. Default 1e-5.",
    )
    args = parser.parse_args()
    args.sync_shift_frames = tuple(int(x.strip()) for x in args.sync_shift_frames.split(",") if x.strip())

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
    if args.preprocessed_dir:
        logger.info(
            "Using precomputed dataset: dir=%s format=%s",
            args.preprocessed_dir,
            args.storage_format,
        )
    full_dataset = LipSyncDataset(
        args.data_dir,
        require_face_detection=not args.no_face_detection,
        preprocessed_dir=args.preprocessed_dir,
        storage_format=args.storage_format,
    )
    if args.no_face_detection and not args.preprocessed_dir:
        logger.warning(
            "⚠️  Face detection disabled - using center crop. This is NOT recommended for production training!"
        )
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    if args.use_augmentation:
        logger.info("Data augmentation enabled for training split")
        train_dataset = AugmentedLipSyncDataset(
            data_dir=args.data_dir,
            base_dataset=full_dataset,
            indices=list(train_subset.indices),
            apply_augmentation=True,
        )
    else:
        train_dataset = ModeOverrideSubset(train_subset, train_mode=True)
    val_dataset = ModeOverrideSubset(val_subset, train_mode=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model = LipSyncModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {param_count:,} parameters")
    logger.info(f"Model moved to device: {next(model.parameters()).device}")

    use_phases = not args.no_freeze
    phase1_end = int(args.freeze_phase1_end)
    phase2_end = int(args.freeze_phase2_end)
    if use_phases:
        logger.info(
            "Phased freezing: Phase 1 (fusion+classifier) epochs 0–%d | "
            "Phase 2 (+audio encoder) %d–%d | Phase 3 (full model) %d+",
            phase1_end - 1, phase1_end, phase2_end - 1, phase2_end,
        )
        logger.info("  Encoder LR when unfrozen: %s", args.lr_encoder)

    # Loss and optimizer (logits)
    criterion = nn.BCEWithLogitsLoss()

    if use_phases:
        _freeze_encoder(model, freeze_visual=True, freeze_audio=True)
        optimizer = torch.optim.Adam(
            _trainable_param_groups(model, 1, args.lr, args.lr_encoder)
        )
        logger.info("Phase 1: training fusion + classifier only (encoders frozen)")
    else:
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
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        if use_phases:
            # Set freeze state and rebuild optimizer for current phase (optimizer state not restored).
            if start_epoch < phase1_end:
                _freeze_encoder(model, freeze_visual=True, freeze_audio=True)
                optimizer = torch.optim.Adam(
                    _trainable_param_groups(model, 1, args.lr, args.lr_encoder)
                )
            elif start_epoch < phase2_end:
                _unfreeze_encoder(model, unfreeze_visual=False, unfreeze_audio=True)
                optimizer = torch.optim.Adam(
                    _trainable_param_groups(model, 2, args.lr, args.lr_encoder)
                )
            else:
                _unfreeze_encoder(model, unfreeze_visual=True, unfreeze_audio=True)
                optimizer = torch.optim.Adam(
                    _trainable_param_groups(model, 3, args.lr, args.lr_encoder)
                )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            logger.info("Resume with phased training: optimizer/scheduler recreated for current phase")
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
        # Phase transitions (phased freezing)
        if use_phases and epoch == phase1_end:
            _unfreeze_encoder(model, unfreeze_visual=False, unfreeze_audio=True)
            optimizer = torch.optim.Adam(
                _trainable_param_groups(model, 2, args.lr, args.lr_encoder)
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            logger.info("Phase 2: unfroze audio encoder (epochs %d–%d)", phase1_end, phase2_end - 1)
        if use_phases and epoch == phase2_end:
            _unfreeze_encoder(model, unfreeze_visual=True, unfreeze_audio=True)
            optimizer = torch.optim.Adam(
                _trainable_param_groups(model, 3, args.lr, args.lr_encoder)
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            logger.info("Phase 3: full model training (epochs %d–%d)", phase2_end, args.epochs - 1)

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_fake_margin=args.contrastive_fake_margin,
            sync_weight=getattr(args, "sync_weight", 0.0),
            sync_shift_frames=getattr(args, "sync_shift_frames", (5, 10, 15)),
            verbose=args.verbose,
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
