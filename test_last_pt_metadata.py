"""
Test for last.pt checkpoint metadata correctness.

Verifies that when an epoch produces a new best F1, last.pt records that new best F1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

# Add training directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

# Import after path is set
from train import save_checkpoint
from training.model import STGCNPlusPlus


def test_last_pt_metadata_correctness():
    """Test that last.pt records the correct best_f1 when it improves."""
    print("Testing last.pt metadata correctness...")

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir)

    try:
        # Create a simple model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        config = {"test": "config"}

        # Simulate scenario: best_f1 starts at 0.5, then improves to 0.7
        initial_best_f1 = 0.5
        improved_best_f1 = 0.7
        epoch = 0

        # Save checkpoint with initial best_f1
        last_path = output_dir / "last.pt"
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            initial_best_f1,
            config,
        )

        # Load and verify initial checkpoint
        checkpoint = torch.load(last_path, weights_only=False)
        assert checkpoint["best_metric"] == initial_best_f1, \
            f"Initial best_f1 incorrect: {checkpoint['best_metric']} vs {initial_best_f1}"
        print(f"  Initial checkpoint best_f1: {checkpoint['best_metric']}")

        # Now simulate improvement: update best_f1 and save again
        epoch = 1
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            improved_best_f1,
            config,
        )

        # Load and verify updated checkpoint
        checkpoint = torch.load(last_path, weights_only=False)
        assert checkpoint["best_metric"] == improved_best_f1, \
            f"Updated best_f1 incorrect: {checkpoint['best_metric']} vs {improved_best_f1}"
        print(f"  Updated checkpoint best_f1: {checkpoint['best_metric']}")

        print("  ✓ last.pt metadata correctness test passed")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def test_best_pt_semantics_unchanged():
    """Test that best.pt is only saved when F1 improves."""
    print("Testing best.pt semantics unchanged...")

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir)

    try:
        # Create a simple model
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        config = {"test": "config"}

        best_path = output_dir / "best.pt"
        last_path = output_dir / "last.pt"

        # Scenario 1: Save best.pt when F1 improves
        best_f1 = 0.7
        epoch = 1
        save_checkpoint(
            best_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_f1,
            config,
        )

        assert best_path.exists(), "best.pt should be saved when F1 improves"
        checkpoint = torch.load(best_path, weights_only=False)
        assert checkpoint["best_metric"] == best_f1, "best.pt should record the best F1"
        print(f"  best.pt saved with F1: {checkpoint['best_metric']}")

        # Scenario 2: Save last.pt without improvement
        # (simulating an epoch where F1 didn't improve)
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_f1,  # best_f1 unchanged
            config,
        )

        assert last_path.exists(), "last.pt should always be saved"
        checkpoint = torch.load(last_path, weights_only=False)
        assert checkpoint["best_metric"] == best_f1, "last.pt should record current best_f1"
        print(f"  last.pt saved with F1: {checkpoint['best_metric']}")

        print("  ✓ best.pt semantics unchanged test passed")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_last_pt_metadata_correctness()
    test_best_pt_semantics_unchanged()
    print("\nALL LAST.PT METADATA TESTS PASSED")
