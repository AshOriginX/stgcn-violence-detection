"""
Analyze checkpoint predictions for validation behavior.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import STGCNDataset
from training.model import STGCNPlusPlus
import numpy as np
from collections import Counter


def analyze_checkpoint_predictions(checkpoint_path):
    """Analyze validation predictions from a checkpoint."""
    print("="*60)
    print(f"CHECKPOINT ANALYSIS: {checkpoint_path}")
    print("="*60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print(f"\nCheckpoint metadata:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best F1: {checkpoint['best_metric']:.4f}")

    # Load validation dataset
    val_dataset = STGCNDataset(
        "outputs/pkl/val.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=True,
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    model = STGCNPlusPlus(num_classes=2, tcn_dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # Collect predictions
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # Calculate metrics
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))

    # Confusion matrix
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    print(f"\nConfusion Matrix:")
    print(f"  Predicted\\Actual    NonFight(0)    Fight(1)")
    print(f"  NonFight(0)         {tn:4d}          {fp:4d}")
    print(f"  Fight(1)            {fn:4d}          {tp:4d}")

    print(f"\nClassification counts:")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")

    print(f"\nPrediction distribution:")
    pred_fight = np.sum(all_predictions == 1)
    pred_nonfight = np.sum(all_predictions == 0)
    print(f"  Predicted Fight: {pred_fight} ({pred_fight/len(all_predictions)*100:.2f}%)")
    print(f"  Predicted NonFight: {pred_nonfight} ({pred_nonfight/len(all_predictions)*100:.2f}%)")

    print(f"\nActual distribution:")
    actual_fight = np.sum(all_labels == 1)
    actual_nonfight = np.sum(all_labels == 0)
    print(f"  Actual Fight: {actual_fight} ({actual_fight/len(all_labels)*100:.2f}%)")
    print(f"  Actual NonFight: {actual_nonfight} ({actual_nonfight/len(all_labels)*100:.2f}%)")

    # Calculate metrics
    accuracy = (tp + tn) / len(all_predictions)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nCalculated metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    # Check for class collapse
    print(f"\nClass collapse analysis:")
    if pred_fight / len(all_predictions) > 0.9:
        print(f"  WARNING: Model predicts >90% Fight - severe class collapse")
    elif pred_fight / len(all_predictions) > 0.8:
        print(f"  WARNING: Model predicts >80% Fight - moderate class collapse")
    elif pred_fight / len(all_predictions) > 0.7:
        print(f"  CAUTION: Model predicts >70% Fight - mild class imbalance")
    else:
        print(f"  No significant class collapse")

    return {
        'confusion_matrix': confusion_matrix,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'pred_fight': pred_fight, 'pred_nonfight': pred_nonfight,
        'actual_fight': actual_fight, 'actual_nonfight': actual_nonfight,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'pred_fight_pct': pred_fight / len(all_predictions),
        'pred_nonfight_pct': pred_nonfight / len(all_predictions),
    }


if __name__ == "__main__":
    # Analyze best checkpoint from phase3_normalized
    best_checkpoint = "outputs/experiment_phase3_normalized/best.pt"
    best_metrics = analyze_checkpoint_predictions(best_checkpoint)

    print("\n" + "="*60)
    print("FINAL EPOCH CHECKPOINT ANALYSIS")
    print("="*60)

    last_checkpoint = "outputs/experiment_phase3_normalized/last.pt"
    last_metrics = analyze_checkpoint_predictions(last_checkpoint)

    print("\n" + "="*60)
    print("COMPARISON: BEST vs FINAL")
    print("="*60)
    print(f"Best F1: {best_metrics['f1']:.4f}")
    print(f"Final F1: {last_metrics['f1']:.4f}")
    print(f"\nBest Fight prediction %: {best_metrics['pred_fight_pct']*100:.2f}%")
    print(f"Final Fight prediction %: {last_metrics['pred_fight_pct']*100:.2f}%")
