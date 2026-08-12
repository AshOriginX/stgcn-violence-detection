"""
Test script for inference setup verification.

Tests:
1. Import all required modules
2. Preprocessing functions work correctly
3. Tensor shapes are correct
4. Normalization statistics are computed correctly
5. Model forward pass works
6. Checkpoint loading works
"""

import sys
import numpy as np
import torch
from pathlib import Path

print("=" * 60)
print("INFERENCE SETUP TESTS")
print("=" * 60)

# Test 1: Import all required modules
print("\n1. Testing imports...")
try:
    from training.preprocessing import (
        WindowIndex,
        handle_invalid_values,
        normalize_persons,
        temporal_resample,
        spatial_normalize,
        generate_windows,
        extract_window,
        preprocess_window,
    )
    print("  ✓ training.preprocessing imports successful")
except Exception as e:
    print(f"  ✗ training.preprocessing import failed: {e}")
    sys.exit(1)

try:
    from training.dataset import STGCNDataset
    print("  ✓ training.dataset imports successful")
except Exception as e:
    print(f"  ✗ training.dataset import failed: {e}")
    sys.exit(1)

try:
    from training.model import STGCNPlusPlus
    print("  ✓ training.model imports successful")
except Exception as e:
    print(f"  ✗ training.model import failed: {e}")
    sys.exit(1)

try:
    from pipeline.detector import YOLODetector
    from pipeline.exporter import PipelineExporter
    from pipeline.extractor import PipelineExtractor
    from pipeline.models import ModelFactory
    from pipeline.pose import RTMPoseEstimator
    from pipeline.tracker import ByteTracker
    print("  ✓ pipeline imports successful")
except Exception as e:
    print(f"  ✗ pipeline import failed: {e}")
    sys.exit(1)

# Test 2: Preprocessing functions work correctly
print("\n2. Testing preprocessing functions...")

# Create synthetic keypoint data
synthetic_keypoint = np.random.randn(2, 200, 17, 2).astype(np.float32)

# Test handle_invalid_values
try:
    clean_keypoint = handle_invalid_values(synthetic_keypoint, invalid_value_policy="zero")
    assert clean_keypoint.shape == synthetic_keypoint.shape
    print("  ✓ handle_invalid_values works")
except Exception as e:
    print(f"  ✗ handle_invalid_values failed: {e}")
    sys.exit(1)

# Test normalize_persons
try:
    normalized = normalize_persons(synthetic_keypoint, target_m=2)
    assert normalized.shape == (2, 200, 17, 2)
    print("  ✓ normalize_persons works")
except Exception as e:
    print(f"  ✗ normalize_persons failed: {e}")
    sys.exit(1)

# Test temporal_resample
try:
    resampled = temporal_resample(synthetic_keypoint, clip_len=150)
    assert resampled.shape == (2, 150, 17, 2)
    print("  ✓ temporal_resample works")
except Exception as e:
    print(f"  ✗ temporal_resample failed: {e}")
    sys.exit(1)

# Test spatial_normalize
try:
    normalized = spatial_normalize(synthetic_keypoint)
    assert normalized.shape == synthetic_keypoint.shape
    # Check that normalization actually changed values
    assert not np.allclose(normalized, synthetic_keypoint)
    print("  ✓ spatial_normalize works")
except Exception as e:
    print(f"  ✗ spatial_normalize failed: {e}")
    sys.exit(1)

# Test generate_windows
try:
    windows = generate_windows(
        keypoint=synthetic_keypoint,
        video_id="test",
        label=0,
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=None,
    )
    assert len(windows) > 0
    assert all(isinstance(w, WindowIndex) for w in windows)
    print(f"  ✓ generate_windows works (generated {len(windows)} windows)")
except Exception as e:
    print(f"  ✗ generate_windows failed: {e}")
    sys.exit(1)

# Test preprocess_window (complete pipeline)
try:
    preprocessed = preprocess_window(
        keypoint=synthetic_keypoint[:, :150, :, :],  # Use exactly 150 frames
        clip_len=150,
        enable_normalization=True,
        invalid_value_policy="zero",
    )
    assert preprocessed.shape == (2, 150, 17, 2)
    print("  ✓ preprocess_window works")
except Exception as e:
    print(f"  ✗ preprocess_window failed: {e}")
    sys.exit(1)

# Test 3: Tensor shapes are correct
print("\n3. Testing tensor shapes...")

# Test final tensor conversion to model format
try:
    preprocessed = preprocess_window(
        keypoint=synthetic_keypoint[:, :150, :, :],
        clip_len=150,
        enable_normalization=True,
        invalid_value_policy="zero",
    )
    # Convert to (C, T, V, M) format
    tensor = torch.from_numpy(preprocessed).float()
    tensor = tensor.permute(3, 1, 2, 0)  # (C, T, V, M)
    assert tensor.shape == (2, 150, 17, 2), f"Expected (2, 150, 17, 2), got {tensor.shape}"
    print(f"  ✓ Tensor shape correct: {tensor.shape}")
except Exception as e:
    print(f"  ✗ Tensor shape test failed: {e}")
    sys.exit(1)

# Test batch format
try:
    batch = tensor.unsqueeze(0)  # Add batch dimension
    assert batch.shape == (1, 2, 150, 17, 2), f"Expected (1, 2, 150, 17, 2), got {batch.shape}"
    print(f"  ✓ Batch shape correct: {batch.shape}")
except Exception as e:
    print(f"  ✗ Batch shape test failed: {e}")
    sys.exit(1)

# Test 4: Normalization statistics
print("\n4. Testing normalization statistics...")

try:
    # Create data with known statistics
    test_keypoint = np.random.randn(2, 150, 17, 2).astype(np.float32) * 10 + 5
    normalized = spatial_normalize(test_keypoint)

    # Check that valid coordinates are normalized (approximately zero mean, unit std)
    valid_mask = ~np.all(test_keypoint == 0, axis=-1)
    if valid_mask.any():
        valid_normalized = normalized[valid_mask]
        mean = valid_normalized.mean(axis=0)
        std = valid_normalized.std(axis=0)
        print(f"  Normalized mean: {mean}, std: {std}")
        assert np.allclose(mean, 0, atol=0.5), f"Mean not close to zero: {mean}"
        assert np.allclose(std, 1, atol=0.5), f"Std not close to one: {std}"
        print("  ✓ Normalization statistics correct")
    else:
        print("  ⚠ No valid coordinates to test normalization")
except Exception as e:
    print(f"  ✗ Normalization statistics test failed: {e}")
    sys.exit(1)

# Test 5: Model forward pass
print("\n5. Testing model forward pass...")

try:
    model = STGCNPlusPlus(num_classes=2, tcn_dropout=0.3)
    model.eval()

    # Create batch
    batch = torch.randn(4, 2, 150, 17, 2)  # Batch of 4

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"
    print(f"  ✓ Model forward pass works: {logits.shape}")

    # Test softmax
    probabilities = torch.softmax(logits, dim=1)
    assert probabilities.shape == (4, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4)), "Probabilities don't sum to 1"
    print(f"  ✓ Softmax works correctly")
except Exception as e:
    print(f"  ✗ Model forward pass test failed: {e}")
    sys.exit(1)

# Test 6: Checkpoint loading
print("\n6. Testing checkpoint loading...")

checkpoint_path = Path("outputs/experiment_phase4_reduce_lr_on_plateau/best.pt")
if not checkpoint_path.exists():
    print(f"  ⚠ Checkpoint not found at {checkpoint_path}")
    print("  Skipping checkpoint loading test")
else:
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"  ✓ Checkpoint loaded successfully")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Best F1: {checkpoint['best_metric']:.4f}")

        # Load model state
        model = STGCNPlusPlus(num_classes=2, tcn_dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"  ✓ Model state loaded successfully")

        # Test forward pass with loaded model
        batch = torch.randn(2, 2, 150, 17, 2)
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (2, 2)
        print(f"  ✓ Loaded model forward pass works")
    except Exception as e:
        print(f"  ✗ Checkpoint loading test failed: {e}")
        sys.exit(1)

# Test 7: STGCNDataset still works with refactored code
print("\n7. Testing STGCNDataset with refactored code...")

try:
    train_pkl = Path("outputs/pkl/train.pkl")
    if train_pkl.exists():
        dataset = STGCNDataset(
            str(train_pkl),
            clip_len=150,
            window_stride=75,
            enable_windowing=True,
            max_windows_per_video=20,
            enable_normalization=True,
        )

        # Test loading a sample
        x, y = dataset[0]
        assert x.shape == (2, 150, 17, 2), f"Expected (2, 150, 17, 2), got {x.shape}"
        assert y.item() in (0, 1)
        print(f"  ✓ STGCNDataset works correctly after refactoring")
        print(f"    Dataset length: {len(dataset)}")
        print(f"    Sample shape: {x.shape}, label: {y.item()}")
    else:
        print(f"  ⚠ train.pkl not found, skipping STGCNDataset test")
except Exception as e:
    print(f"  ✗ STGCNDataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
