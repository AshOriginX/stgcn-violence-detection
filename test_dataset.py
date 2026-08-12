"""
Test script for STGCNDataset implementation.

This script validates all requirements from the dataset specification.
"""

import torch
from torch.utils.data import DataLoader
from training.dataset import STGCNDataset
from training.model import STGCNPlusPlus


def test_dataset():
    """Test dataset loading and validation."""
    print("=" * 60)
    print("Testing STGCNDataset")
    print("=" * 60)

    # Load datasets
    train = STGCNDataset("outputs/pkl/train.pkl", clip_len=150, train=False)
    val = STGCNDataset("outputs/pkl/val.pkl", clip_len=150, train=False)

    # Test lengths
    print(f"\n1. Dataset lengths:")
    print(f"   train: {len(train)} (expected: 3909)")
    print(f"   val: {len(val)} (expected: 978)")

    assert len(train) == 3909, f"Train length mismatch: {len(train)} != 3909"
    assert len(val) == 978, f"Val length mismatch: {len(val)} != 978"
    print("   ✓ Lengths correct")

    # Test sample shapes
    print(f"\n2. Sample shapes:")
    test_indices = [0, 1, len(train) // 2, len(train) - 1]
    for idx in test_indices:
        x, y = train[idx]
        print(f"   train[{idx}]: x.shape={x.shape}, y={y}, y.dtype={y.dtype}")
        assert x.shape == (2, 150, 17, 2), f"Shape mismatch: {x.shape}"
        assert x.dtype == torch.float32, f"dtype mismatch: {x.dtype}"
        assert y.dtype == torch.long, f"label dtype mismatch: {y.dtype}"
        assert y.item() in (0, 1), f"Invalid label: {y.item()}"
        assert torch.isfinite(x).all(), f"Non-finite values in sample {idx}"
    print("   ✓ Shapes and dtypes correct")

    # Test validation determinism
    print(f"\n3. Validation determinism:")
    x1, y1 = val[0]
    x2, y2 = val[0]
    assert torch.equal(x1, x2), "Validation samples not identical"
    assert torch.equal(y1, y2), "Validation labels not identical"
    print("   ✓ Validation is deterministic")

    # Test DataLoader
    print(f"\n4. DataLoader batch generation:")
    train_loader = DataLoader(
        train, batch_size=2, shuffle=True, num_workers=0
    )
    x_batch, y_batch = next(iter(train_loader))
    print(f"   batch: x.shape={x_batch.shape}, y.shape={y_batch.shape}")
    assert x_batch.shape == (2, 2, 150, 17, 2), f"Batch shape mismatch: {x_batch.shape}"
    assert y_batch.shape == (2,), f"Label batch shape mismatch: {y_batch.shape}"
    print("   ✓ DataLoader works correctly")

    # Test model forward pass
    print(f"\n5. Model forward pass:")
    model = STGCNPlusPlus(num_classes=2)
    logits = model(x_batch)
    print(f"   logits.shape={logits.shape}")
    assert logits.shape == (2, 2), f"Logits shape mismatch: {logits.shape}"
    print("   ✓ Model forward pass works")

    # Test training step
    print(f"\n6. Training smoke test:")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    print(f"   loss: {float(loss):.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    print(f"   parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   gradients finite: {all(torch.isfinite(p.grad).all() if p.grad is not None else True for p in model.parameters())}")
    print("   ✓ Training step successful")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()
