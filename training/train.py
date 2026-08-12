"""
Production-quality training script for ST-GCN++ Violence Detection.

This script provides a complete training pipeline with:
- Data loading and validation
- Model training with mixed precision
- Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
- Checkpointing and resume support
- Reproducibility controls
- CLI interface
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import STGCNDataset
from training.model import STGCNPlusPlus


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    deterministic : bool, optional
        Whether to use deterministic CUDA algorithms. Default is False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate classification metrics for binary classification.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted logits with shape (N, 2).
    targets : torch.Tensor
        Ground truth labels with shape (N,).

    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy, precision, recall, F1, and confusion matrix.
    """
    pred_labels = predictions.argmax(dim=1)
    correct = (pred_labels == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total

    # Confusion matrix
    tp = ((pred_labels == 1) & (targets == 1)).sum().item()
    tn = ((pred_labels == 0) & (targets == 0)).sum().item()
    fp = ((pred_labels == 1) & (targets == 0)).sum().item()
    fn = ((pred_labels == 0) & (targets == 1)).sum().item()

    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    dataloader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        Device to train on.
    scaler : torch.GradScaler or None
        GradScaler for mixed precision training.
    epoch : int
        Current epoch number.
    max_grad_norm : float, optional
        Maximum gradient norm for clipping. Default is 1.0.

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        pred_labels = logits.argmax(dim=1)
        correct += (pred_labels == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate the model.

    Parameters
    ----------
    model : nn.Module
        Model to validate.
    dataloader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device to validate on.

    Returns
    -------
    Dict[str, float]
        Dictionary containing loss and all metrics.
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        all_predictions.append(logits.cpu())
        all_targets.append(y.cpu())

    avg_loss = total_loss / len(dataloader)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_predictions, all_targets)
    metrics["loss"] = avg_loss

    return metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_metric: float,
    config: Dict,
) -> None:
    """
    Save training checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to save checkpoint.
    model : nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Scheduler to save.
    epoch : int
        Current epoch.
    best_metric : float
        Best validation metric.
    config : Dict
        Training configuration.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": config,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float, Dict]:
    """
    Load training checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint.
    model : nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer
        Optimizer to load state into.
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Scheduler to load state into.

    Returns
    -------
    Tuple[int, float, Dict]
        Epoch, best metric, and configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_metric = checkpoint["best_metric"]
    config = checkpoint.get("config", {})

    return epoch, best_metric, config


def load_pretrained_weights(
    checkpoint_path: Path,
    model: nn.Module,
) -> Dict:
    """Load only model weights from a checkpoint (for fine-tuning)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("config", {})


def run_smoke_test(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> None:
    """
    Run first-batch smoke test before training.

    Verifies forward pass, loss computation, backward pass, and gradient computation
    without updating model weights or optimizer state.

    Parameters
    ----------
    model : nn.Module
        Model to test.
    dataloader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        Device to test on.
    scaler : torch.GradScaler or None
        GradScaler for mixed precision.

    Raises
    ------
    RuntimeError
        If smoke test fails.
    """
    print("Running first-batch smoke test...")
    model.train()

    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    if scaler is not None:
        with torch.amp.autocast(device_type="cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        # Unscale gradients to check them, but do NOT step optimizer
        scaler.unscale_(optimizer)
    else:
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

    assert torch.isfinite(loss), "Loss is not finite"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"

    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Gradients contain non-finite values"

    # DO NOT call optimizer.step() or scaler.step()
    # DO NOT update model weights or optimizer state
    print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Train ST-GCN++ Violence Detection Model"
    )

    # Data paths
    parser.add_argument(
        "--train-pkl",
        type=str,
        default="outputs/pkl/train.pkl",
        help="Path to training PKL file",
    )
    parser.add_argument(
        "--val-pkl",
        type=str,
        default="outputs/pkl/val.pkl",
        help="Path to validation PKL file",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms",
    )

    # Device and AMP
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints",
        help="Output directory for checkpoints",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to checkpoint for fine-tuning (loads model weights only)",
    )
    parser.add_argument(
        "--epoch-sleep",
        type=float,
        default=0,
        help="Seconds to sleep after each completed epoch",
    )
    parser.add_argument(
        "--max-windows-per-video",
        type=int,
        default=20,
        help="Maximum windows to extract per source video (None = unlimited)",
    )
    parser.add_argument(
        "--tcn-dropout",
        type=float,
        default=0.3,
        help="TCN dropout rate",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--enable-normalization",
        action="store_true",
        default=True,
        help="Enable spatial normalization of coordinates",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Disable spatial normalization of coordinates",
    )

    args = parser.parse_args()

    # Handle AMP flag
    use_amp = args.amp and not args.no_amp
    if args.no_amp:
        use_amp = False

    # Handle normalization flag
    enable_normalization = args.enable_normalization and not args.no_normalization

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set seed
    set_seed(args.seed, args.deterministic)
    print(f"Seed: {args.seed}, Deterministic: {args.deterministic}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = STGCNDataset(
        args.train_pkl,
        clip_len=150,
        window_stride=75,
        train=True,
        invalid_value_policy="zero",
        enable_windowing=True,
        max_windows_per_video=args.max_windows_per_video,
        enable_normalization=enable_normalization,
    )
    val_dataset = STGCNDataset(
        args.val_pkl,
        clip_len=150,
        window_stride=75,
        train=False,
        invalid_value_policy="zero",
        enable_windowing=True,
        max_windows_per_video=args.max_windows_per_video,
        enable_normalization=enable_normalization,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Generate dataset reports
    print("\n=== DATASET REPORT ===")
    train_report = train_dataset.generate_report()
    val_report = val_dataset.generate_report()

    print(f"\nTrain dataset:")
    print(f"  Original annotations: {train_report['original_annotation_count']}")
    print(f"  Windows before capping: {train_report['windows_before_capping']}")
    print(f"  Resulting windows: {train_report['resulting_window_count']}")
    print(f"  Videos capped: {train_report['videos_capped']}")
    print(f"  Windows removed: {train_report['windows_removed']}")
    print(f"  Windows by dataset: {train_report['windows_by_dataset']}")
    print(f"  Windows by label: {train_report['windows_by_label']}")
    print(f"  Min windows per video: {train_report['min_windows_per_video']}")
    print(f"  Max windows per video: {train_report['max_windows_per_video']}")
    print(f"  Mean windows per video: {train_report['mean_windows_per_video']:.2f}")
    print(f"  Top 20 videos by window count:")
    for video_id, count in train_report['top_20_videos_by_window_count']:
        print(f"    {video_id}: {count}")

    print(f"\nVal dataset:")
    print(f"  Original annotations: {val_report['original_annotation_count']}")
    print(f"  Windows before capping: {val_report['windows_before_capping']}")
    print(f"  Resulting windows: {val_report['resulting_window_count']}")
    print(f"  Videos capped: {val_report['videos_capped']}")
    print(f"  Windows removed: {val_report['windows_removed']}")
    print(f"  Windows by dataset: {val_report['windows_by_dataset']}")
    print(f"  Windows by label: {val_report['windows_by_label']}")
    print(f"  Min windows per video: {val_report['min_windows_per_video']}")
    print(f"  Max windows per video: {val_report['max_windows_per_video']}")
    print(f"  Mean windows per video: {val_report['mean_windows_per_video']:.2f}")

    # Check train/val video-ID overlap
    train_video_ids = set(window.video_id for window in train_dataset.windows)
    val_video_ids = set(window.video_id for window in val_dataset.windows)
    overlap = train_video_ids & val_video_ids
    print(f"\nTrain/Val video-ID overlap: {len(overlap)}")
    if overlap:
        print(f"  WARNING: Overlapping video IDs: {sorted(overlap)[:10]}")
    else:
        print(f"  ✓ No overlap (correct)")

    # Sanity check
    print("Running dataset sanity check...")
    x, y = train_dataset[0]
    assert x.shape == (2, 150, 17, 2), f"Unexpected shape: {x.shape}"
    assert y.item() in (0, 1), f"Unexpected label: {y.item()}"
    print("Sanity check passed.")

    # Create data loaders
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Create model
    print("Creating model...")
    model = STGCNPlusPlus(num_classes=2, tcn_dropout=args.tcn_dropout).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # Create GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    print(f"AMP: {scaler is not None}")

    # Temporarily disable AMP for smoke test to isolate issues
    smoke_test_scaler = None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_epoch, best_f1, _ = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
        )
        start_epoch += 1  # Continue from next epoch
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    elif args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}...")
        load_pretrained_weights(Path(args.pretrained), model)
        print("Pretrained weights loaded (optimizer reset for fine-tuning)")

    # Save configuration
    config = vars(args)
    # Add window_stride to config (used in dataset but not CLI arg)
    config["window_stride"] = 75
    # Add enable_normalization to config (derived from args)
    config["enable_normalization"] = enable_normalization
    # Add scheduler configuration
    config["scheduler"] = {
        "type": "ReduceLROnPlateau",
        "mode": "max",
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-6,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Run smoke test
    run_smoke_test(model, train_loader, criterion, optimizer, device, smoke_test_scaler)

    # Training loop
    print("\nStarting training...")
    best_path = output_dir / "best.pt"
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            args.max_grad_norm,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print(
            f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}"
        )
        print(
            f"  Val:   loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"precision={val_metrics['precision']:.4f}, "
            f"recall={val_metrics['recall']:.4f}, "
            f"F1={val_metrics['f1']:.4f}"
        )

        # Update best_f1 if current validation F1 improves
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                config,
            )
            print(f"  New best model saved (F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{args.patience} epochs")

        # Update scheduler with validation F1 (after best_f1 update)
        scheduler.step(val_metrics["f1"])

        # Save last checkpoint with updated best_f1
        last_path = output_dir / "last.pt"
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_f1,
            config,
        )

        # Early stopping check
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

    print("\nTraining completed.")
    print(f"Best validation F1: {best_f1:.4f}")

    # Restore best checkpoint to in-memory model
    if best_path.exists():
        print(f"Restoring best checkpoint to in-memory model...")
        load_checkpoint(best_path, model, optimizer, scheduler)
        print("Best checkpoint restored.")


if __name__ == "__main__":
    main()
