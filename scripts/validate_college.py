"""
Validate checkpoint on college_violence val set with threshold analysis.

Reports precision, recall, F1, FP, FN at multiple thresholds.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.dataset import STGCNDataset
from training.model import STGCNPlusPlus


def evaluate_at_threshold(fight_probs, labels, threshold, min_consecutive=2):
    """Video-level evaluation using temporal consecutive-window logic."""
    # Group by video (approximate: use window labels directly for window-level)
    preds = (fight_probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate on college val set")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--val-pkl", type=Path, default=Path("outputs/pkl/college/val.pkl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/college_validation.json")
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.val_pkl.exists():
        print(f"Val PKL not found: {args.val_pkl}")
        sys.exit(1)

    val_dataset = STGCNDataset(
        args.val_pkl,
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = STGCNPlusPlus(num_classes=2, tcn_dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy())

    fight_probs = np.array(all_probs)
    labels = np.array(all_labels)

    print("=" * 60)
    print(f"COLLEGE VALIDATION: {args.checkpoint}")
    print("=" * 60)
    print(f"Windows: {len(fight_probs)}")
    print(f"Fight windows (label=1): {(labels == 1).sum()}")
    print(f"NonFight windows (label=0): {(labels == 0).sum()}")
    print(f"\nProbability distribution:")
    print(f"  Mean: {fight_probs.mean():.4f}")
    print(f"  Std:  {fight_probs.std():.4f}")
    print(f"  Min:  {fight_probs.min():.4f}")
    print(f"  Max:  {fight_probs.max():.4f}")
    for label_val, name in [(0, "NonFight"), (1, "Fight")]:
        mask = labels == label_val
        if mask.any():
            print(f"  {name} mean prob: {fight_probs[mask].mean():.4f}")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'FP':>5} {'FN':>5}")
    print("-" * 55)
    for t in thresholds:
        r = evaluate_at_threshold(fight_probs, labels, t)
        results.append(r)
        print(
            f"{r['threshold']:>10.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1']:>8.4f} {r['fp']:>5} {r['fn']:>5}"
        )

    # Best threshold for FP reduction (precision >= 0.8 or lowest FP)
    best_fp = min(results, key=lambda r: (r["fp"], -r["recall"]))
    print(f"\nLowest-FP threshold: {best_fp['threshold']:.2f} "
          f"(FP={best_fp['fp']}, recall={best_fp['recall']:.4f})")

    output = {
        "checkpoint": str(args.checkpoint),
        "val_pkl": str(args.val_pkl),
        "num_windows": len(fight_probs),
        "prob_distribution": {
            "mean": float(fight_probs.mean()),
            "std": float(fight_probs.std()),
            "min": float(fight_probs.min()),
            "max": float(fight_probs.max()),
        },
        "threshold_results": results,
        "best_fp_threshold": best_fp,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
