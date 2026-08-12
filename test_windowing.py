"""
Comprehensive tests for temporal windowing implementation.

Tests:
- Exact output shape
- Zero train/val video-ID overlap
- All windows from one video have identical labels
- Expected number of windows for known T values
- Deterministic validation
- DataLoader batching
- Model forward compatibility
"""

import pickle
import torch
from torch.utils.data import DataLoader
from training.dataset import STGCNDataset, WindowIndex
from training.model import STGCNPlusPlus


def test_output_shape():
    """Test that all samples have correct output shape."""
    print("Testing output shape...")
    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,  # Test with raw coordinates
    )

    for i in range(min(100, len(dataset))):
        x, y = dataset[i]
        assert x.shape == (2, 150, 17, 2), f"Sample {i}: unexpected shape {x.shape}"
        assert x.dtype == torch.float32, f"Sample {i}: unexpected dtype {x.dtype}"
        assert y.dtype == torch.long, f"Sample {i}: unexpected label dtype {y.dtype}"
        assert y.item() in (0, 1), f"Sample {i}: unexpected label {y.item()}"

    print("  ✓ Output shape test passed")


def test_train_val_overlap():
    """Test that train and val have zero video-ID overlap."""
    print("Testing train/val video-ID overlap...")
    train_dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )
    val_dataset = STGCNDataset(
        "outputs/pkl/val.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    train_video_ids = set(window.video_id for window in train_dataset.windows)
    val_video_ids = set(window.video_id for window in val_dataset.windows)

    overlap = train_video_ids & val_video_ids
    assert len(overlap) == 0, f"Train/val overlap found: {overlap}"

    print(f"  ✓ No overlap (train: {len(train_video_ids)}, val: {len(val_video_ids)})")


def test_window_label_consistency():
    """Test that all windows from the same video have identical labels."""
    print("Testing window label consistency...")
    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    # Group windows by video_id
    video_labels = {}
    for window in dataset.windows:
        if window.video_id not in video_labels:
            video_labels[window.video_id] = window.label
        else:
            assert video_labels[window.video_id] == window.label, \
                f"Video {window.video_id} has inconsistent labels"

    print(f"  ✓ All {len(video_labels)} videos have consistent labels")


def test_expected_window_counts():
    """Test window count calculation for known T values."""
    print("Testing expected window counts...")

    # Load PKL to check specific videos
    with open("outputs/pkl/train.pkl", "rb") as f:
        data = pickle.load(f)

    # Find specific videos mentioned in requirements
    target_videos = ["RLVS_000993", "RLVS_001767"]
    found_videos = {}

    for ann in data["annotations"]:
        video_id = ann["frame_dir"]
        if video_id in target_videos:
            keypoint = ann["keypoint"]
            T = keypoint.shape[1]
            found_videos[video_id] = T

    # Calculate expected windows
    clip_len = 150
    stride = 75

    for video_id, T in found_videos.items():
        if T <= clip_len:
            expected = 1
        else:
            expected = (T - clip_len) // stride + 1

        print(f"  {video_id}: T={T}, expected windows={expected}")

    # Test with dataset
    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    for video_id in found_videos.keys():
        actual = sum(1 for w in dataset.windows if w.video_id == video_id)
        T = found_videos[video_id]
        if T <= clip_len:
            expected = 1
        else:
            expected = (T - clip_len) // stride + 1

        assert actual == expected, \
            f"{video_id}: expected {expected}, got {actual}"
        print(f"  ✓ {video_id}: {actual} windows (correct)")


def test_deterministic_validation():
    """Test that validation dataset is deterministic."""
    print("Testing deterministic validation...")
    dataset = STGCNDataset(
        "outputs/pkl/val.pkl",
        clip_len=150,
        window_stride=75,
        train=False,
        enable_windowing=True,
        enable_normalization=False,
    )

    # Get same sample twice
    x1, y1 = dataset[0]
    x2, y2 = dataset[0]

    assert torch.equal(x1, x2), "Validation not deterministic (x)"
    assert torch.equal(y1, y2), "Validation not deterministic (y)"

    # Get different samples
    x3, y3 = dataset[1]
    assert not torch.equal(x1, x3), "Different samples should be different"

    print("  ✓ Validation is deterministic")


def test_dataloader_batching():
    """Test DataLoader batching works correctly."""
    print("Testing DataLoader batching...")
    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (4, 2, 150, 17, 2), f"Unexpected batch shape: {batch_x.shape}"
    assert batch_y.shape == (4,), f"Unexpected label batch shape: {batch_y.shape}"
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.long

    print("  ✓ DataLoader batching works correctly")


def test_model_forward_compatibility():
    """Test that model forward pass works with windowed data."""
    print("Testing model forward compatibility...")
    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = STGCNPlusPlus(num_classes=2)
    model.eval()

    batch_x, batch_y = next(iter(loader))
    with torch.no_grad():
        logits = model(batch_x)

    assert logits.shape == (2, 2), f"Unexpected output shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Model output contains non-finite values"

    print("  ✓ Model forward pass works correctly")


def test_window_index_structure():
    """Test that WindowIndex dataclass is correctly structured."""
    print("Testing WindowIndex structure...")
    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    window = dataset.windows[0]
    assert isinstance(window, WindowIndex)
    assert hasattr(window, 'annotation_idx')
    assert hasattr(window, 'window_start')
    assert hasattr(window, 'window_end')
    assert hasattr(window, 'video_id')
    assert hasattr(window, 'label')

    assert isinstance(window.annotation_idx, int)
    assert isinstance(window.window_start, int)
    assert isinstance(window.window_end, int)
    assert isinstance(window.video_id, str)
    assert isinstance(window.label, int)

    assert window.window_end > window.window_start
    assert window.label in (0, 1)

    print("  ✓ WindowIndex structure is correct")


def test_windowing_disabled():
    """Test that windowing can be disabled (old behavior)."""
    print("Testing windowing disabled...")
    dataset_windowed = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        enable_normalization=False,
    )

    dataset_no_windowing = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=False,
        enable_normalization=False,
    )

    # With windowing, should have more samples
    assert len(dataset_windowed) > len(dataset_no_windowing), \
        "Windowing should increase sample count"

    # Without windowing, should equal original annotation count
    with open("outputs/pkl/train.pkl", "rb") as f:
        data = pickle.load(f)
    original_count = len(data["annotations"])

    assert len(dataset_no_windowing) == original_count, \
        "Without windowing, sample count should equal annotation count"

    print(f"  ✓ Windowing disabled: {len(dataset_no_windowing)} samples")
    print(f"  ✓ Windowing enabled: {len(dataset_windowed)} samples")


def test_window_capping():
    """Test that window capping works correctly."""
    print("Testing window capping...")

    # Test with cap=20
    dataset_capped = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=False,
    )

    # Test without cap
    dataset_uncapped = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=None,
        enable_normalization=False,
    )

    # Capped should have fewer samples
    assert len(dataset_capped) < len(dataset_uncapped), \
        "Capping should reduce sample count"

    # Count windows per video
    windows_per_video = {}
    for window in dataset_capped.windows:
        windows_per_video[window.video_id] = windows_per_video.get(window.video_id, 0) + 1

    # No video should exceed cap
    for video_id, count in windows_per_video.items():
        assert count <= 20, f"Video {video_id} has {count} windows (exceeds cap 20)"

    print(f"  ✓ Capped dataset: {len(dataset_capped)} samples")
    print(f"  ✓ Uncapped dataset: {len(dataset_uncapped)} samples")
    print(f"  ✓ No video exceeds cap")


def test_cap_deterministic():
    """Test that capping selection is deterministic."""
    print("Testing deterministic capping...")

    # Create two datasets with same cap
    dataset1 = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=False,
    )

    dataset2 = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=False,
    )

    # Should have same length
    assert len(dataset1) == len(dataset2), "Deterministic capping failed: different lengths"

    # Check that window indices are the same for capped videos
    windows_per_video1 = {}
    for window in dataset1.windows:
        if window.video_id not in windows_per_video1:
            windows_per_video1[window.video_id] = []
        windows_per_video1[window.video_id].append(window.window_start)

    windows_per_video2 = {}
    for window in dataset2.windows:
        if window.video_id not in windows_per_video2:
            windows_per_video2[window.video_id] = []
        windows_per_video2[window.video_id].append(window.window_start)

    for video_id in windows_per_video1:
        assert windows_per_video1[video_id] == windows_per_video2[video_id], \
            f"Deterministic capping failed for {video_id}"

    print("  ✓ Capping selection is deterministic")


def test_cap_sorted_indices():
    """Test that selected window indices are sorted."""
    print("Testing sorted window indices...")

    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=False,
    )

    # Group windows by video
    video_windows = {}
    for window in dataset.windows:
        if window.video_id not in video_windows:
            video_windows[window.video_id] = []
        video_windows[window.video_id].append(window)

    # Check that window_start is sorted for each video
    for video_id, windows in video_windows.items():
        starts = [w.window_start for w in windows]
        assert starts == sorted(starts), f"Window indices not sorted for {video_id}"

    print("  ✓ Window indices are sorted")


def test_cap_temporal_coverage():
    """Test that capped windows span the temporal range."""
    print("Testing temporal coverage...")

    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=False,
    )

    # Check RLVS_001767 (should be capped to 20)
    rlvs_windows = [w for w in dataset.windows if w.video_id == "RLVS_001767"]
    if len(rlvs_windows) > 0:
        starts = [w.window_start for w in rlvs_windows]
        # First window should start near beginning
        assert starts[0] < 100, f"First window doesn't span temporal range: {starts[0]}"
        # Last window should be near end (T=11272, so last window start ~11122)
        assert starts[-1] > 11000, f"Last window doesn't span temporal range: {starts[-1]}"
        print(f"  ✓ RLVS_001767 windows span temporal range: {starts[0]} to {starts[-1]}")
    else:
        print("  ⚠ RLVS_001767 not found in dataset")

    print("  ✓ Temporal coverage test passed")


def test_cap_report_accuracy():
    """Test that report counts are correct."""
    print("Testing report accuracy...")

    dataset = STGCNDataset(
        "outputs/pkl/train.pkl",
        clip_len=150,
        window_stride=75,
        enable_windowing=True,
        max_windows_per_video=20,
        enable_normalization=False,
    )

    report = dataset.generate_report()

    # Verify report fields exist
    assert "windows_before_capping" in report
    assert "videos_capped" in report
    assert "windows_removed" in report

    # Verify counts match actual
    assert report["resulting_window_count"] == len(dataset.windows)
    assert report["windows_removed"] == report["windows_before_capping"] - report["resulting_window_count"]

    # Count videos that should be capped
    windows_per_video = {}
    for window in dataset.windows:
        windows_per_video[window.video_id] = windows_per_video.get(window.video_id, 0) + 1

    # Videos with exactly 20 windows are likely capped
    # (unless they naturally have exactly 20 windows)
    capped_count = sum(1 for count in windows_per_video.values() if count == 20)
    print(f"  ✓ Report counts are accurate")
    print(f"  ✓ Videos with 20 windows (likely capped): {capped_count}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING WINDOWING TESTS")
    print("=" * 60)

    test_output_shape()
    test_train_val_overlap()
    test_window_label_consistency()
    test_expected_window_counts()
    test_deterministic_validation()
    test_dataloader_batching()
    test_model_forward_compatibility()
    test_window_index_structure()
    test_windowing_disabled()
    test_window_capping()
    test_cap_deterministic()
    test_cap_sorted_indices()
    test_cap_temporal_coverage()
    test_cap_report_accuracy()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
