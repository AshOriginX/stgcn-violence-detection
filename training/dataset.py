"""
Production-quality PyTorch Dataset for ST-GCN++ Violence Detection.

This module provides a robust dataset loader for the preprocessed PKL annotations
with deterministic temporal normalization, validation, memory efficiency, and
temporal windowing for long videos.

KNOWN LIMITATION:
When extracting windows from long videos, every window inherits the video-level
label. This means windows from Fight videos may not actually contain fighting,
and windows from NonFight videos may contain fighting. This is a weak-label
problem inherent to video-level annotation with temporal windowing.
"""

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

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


class STGCNDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ST-GCN++ Violence Detection with temporal windowing.

    This dataset loads preprocessed PKL annotations and returns samples with
    normalized temporal dimension (T=150) for consistent model input.

    For videos with T > clip_len, multiple overlapping windows are extracted
    with stride = clip_len // 2 (50% overlap). All windows from the same video
    inherit the video-level label.

    Parameters
    ----------
    pkl_path : str or Path
        Path to the PKL file containing annotations (train.pkl or val.pkl).
    clip_len : int, optional
        Target temporal length for all samples. Default is 150.
    window_stride : int, optional
        Stride for window extraction. Default is 75 (50% of clip_len).
    train : bool, optional
        Whether this is the training split. Used for potential augmentation.
        Currently no augmentation is implemented. Default is False.
    invalid_value_policy : str, optional
        Policy for handling NaN/Inf values. Options: "zero" (replace with 0)
        or "error" (raise exception). Default is "zero".
    enable_windowing : bool, optional
        Whether to enable temporal windowing for long videos. Default is True.

    Attributes
    ----------
    annotations : list
        List of annotation dictionaries from the PKL file.
    clip_len : int
       Target temporal length.
    window_stride : int
        Stride for window extraction.
    train : bool
        Training mode flag.
    invalid_value_policy : str
        Policy for invalid values.
    windows : list
        List of WindowIndex objects for all samples.
    """

    def __init__(
        self,
        pkl_path: str | Path,
        clip_len: int = 150,
        window_stride: int = 75,
        train: bool = False,
        invalid_value_policy: Literal["zero", "error"] = "zero",
        enable_windowing: bool = True,
        max_windows_per_video: Optional[int] = None,
        enable_normalization: bool = True,
    ):
        self.pkl_path = Path(pkl_path)
        self.clip_len = clip_len
        self.window_stride = window_stride
        self.train = train
        self.invalid_value_policy = invalid_value_policy
        self.enable_windowing = enable_windowing
        self.max_windows_per_video = max_windows_per_video
        self.enable_normalization = enable_normalization

        if not self.pkl_path.exists():
            raise FileNotFoundError(f"PKL file not found: {self.pkl_path}")

        self._load_pkl()
        self._build_window_index()

    def _load_pkl(self) -> None:
        """Load annotations from PKL file."""
        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)

        if "annotations" not in data:
            raise ValueError(f"PKL file missing 'annotations' key: {self.pkl_path}")

        self.annotations = data["annotations"]

    def _build_window_index(self) -> None:
        """
        Build window index for all annotations.

        For each annotation:
        - If T == clip_len: create one window covering the entire video
        - If T < clip_len: create one window (will be padded later)
        - If T > clip_len and enable_windowing: create overlapping windows with stride
        - If T > clip_len and not enable_windowing: uniformly sample to clip_len (old behavior)

        If max_windows_per_video is set and a video exceeds the cap,
        select windows deterministically and uniformly across the temporal range.
        """
        self.windows = []
        self._videos_capped = 0
        self._windows_before_capping = 0

        for ann_idx, annotation in enumerate(self.annotations):
            keypoint = annotation["keypoint"]
            video_id = annotation["frame_dir"]
            label = annotation["label"]

            # Generate all windows for this video using extracted function
            video_windows = generate_windows(
                keypoint=keypoint,
                video_id=video_id,
                label=label,
                clip_len=self.clip_len,
                window_stride=self.window_stride,
                enable_windowing=self.enable_windowing,
                max_windows_per_video=self.max_windows_per_video,
            )

            # Update annotation_idx to match current annotation index
            for window in video_windows:
                window.annotation_idx = ann_idx

            # Track windows before capping
            windows_before_cap = len(video_windows)

            if self.max_windows_per_video is not None and len(video_windows) == self.max_windows_per_video:
                self._videos_capped += 1

            self._windows_before_capping += windows_before_cap
            self.windows.extend(video_windows)

    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the window to retrieve.

        Returns
        -------
        x : torch.Tensor
            Keypoint tensor with shape (2, 150, 17, 2) and dtype torch.float32.
        y : torch.Tensor
            Label tensor with shape () and dtype torch.long.
        """
        window = self.windows[index]
        annotation = self.annotations[window.annotation_idx]

        # Extract keypoint and apply temporal window
        keypoint = annotation["keypoint"]
        keypoint = keypoint[:, window.window_start:window.window_end, :, :]

        # Validate keypoint shape
        self._validate_keypoint(keypoint, index)

        # Extract label from window (same as video-level label)
        label = window.label
        self._validate_label(label, index)

        # Apply complete preprocessing pipeline using extracted function
        keypoint = preprocess_window(
            keypoint=keypoint,
            clip_len=self.clip_len,
            enable_normalization=self.enable_normalization,
            invalid_value_policy=self.invalid_value_policy,
        )

        # Convert to tensor and permute to (C, T, V, M)
        # From (M, T, V, C) to (C, T, V, M)
        keypoint = torch.from_numpy(keypoint).float()
        keypoint = keypoint.permute(3, 1, 2, 0)  # (C, T, V, M)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return keypoint, label

    def _validate_keypoint(self, keypoint: np.ndarray, index: int) -> None:
        """
        Validate keypoint array structure and values.

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoint array to validate.
        index : int
            Annotation index for error reporting.

        Raises
        ------
        ValueError
            If keypoint structure is invalid.
        """
        if not isinstance(keypoint, np.ndarray):
            raise ValueError(
                f"Annotation {index}: keypoint must be numpy array, got {type(keypoint)}"
            )

        if keypoint.ndim != 4:
            raise ValueError(
                f"Annotation {index}: keypoint must have 4 dimensions (M,T,V,C), "
                f"got {keypoint.ndim} dimensions with shape {keypoint.shape}"
            )

        M, T, V, C = keypoint.shape

        if C != 2:
            raise ValueError(
                f"Annotation {index}: keypoint must have 2 coordinate dimensions (x,y), "
                f"got {C} with shape {keypoint.shape}"
            )

        if V != 17:
            raise ValueError(
                f"Annotation {index}: keypoint must have 17 keypoints (COCO format), "
                f"got {V} with shape {keypoint.shape}"
            )

        if M > 2:
            raise ValueError(
                f"Annotation {index}: keypoint has {M} persons, "
                f"max allowed is 2. Shape: {keypoint.shape}"
            )

        if T < 1:
            raise ValueError(
                f"Annotation {index}: keypoint has {T} frames, "
                f"minimum is 1. Shape: {keypoint.shape}"
            )

    def _validate_label(self, label: int, index: int) -> None:
        """
        Validate label value.

        Parameters
        ----------
        label : int
            Label value to validate.
        index : int
            Annotation index for error reporting.

        Raises
        ------
        ValueError
            If label is not 0 or 1.
        """
        if label not in (0, 1):
            raise ValueError(
                f"Annotation {index}: label must be 0 (NonFight) or 1 (Fight), "
                f"got {label}"
            )

    def generate_report(self) -> dict:
        """
        Generate a detailed report about the dataset.

        Returns
        -------
        dict
            Dictionary containing dataset statistics.
        """
        original_count = len(self.annotations)
        window_count = len(self.windows)

        # Count windows by video_id
        windows_per_video = Counter()
        for window in self.windows:
            windows_per_video[window.video_id] += 1

        # Extract dataset names from video_ids
        dataset_counts = Counter()
        label_counts = Counter()
        dataset_label_counts = Counter()

        for window in self.windows:
            # Extract dataset name from video_id (e.g., "RLVS_000993" -> "RLVS")
            dataset_name = window.video_id.split("_")[0] if "_" in window.video_id else "unknown"
            dataset_counts[dataset_name] += 1
            label_counts[window.label] += 1
            dataset_label_counts[(dataset_name, window.label)] += 1

        # Statistics for windows per video
        windows_per_video_values = list(windows_per_video.values())
        min_windows = min(windows_per_video_values) if windows_per_video_values else 0
        max_windows = max(windows_per_video_values) if windows_per_video_values else 0
        mean_windows = np.mean(windows_per_video_values) if windows_per_video_values else 0

        # Top 20 videos by window count
        top_videos = windows_per_video.most_common(20)

        return {
            "original_annotation_count": original_count,
            "resulting_window_count": window_count,
            "windows_before_capping": self._windows_before_capping,
            "videos_capped": self._videos_capped,
            "windows_removed": self._windows_before_capping - window_count,
            "windows_by_dataset": dict(dataset_counts),
            "windows_by_label": dict(label_counts),
            "windows_by_dataset_label": dict(dataset_label_counts),
            "min_windows_per_video": min_windows,
            "max_windows_per_video": max_windows,
            "mean_windows_per_video": mean_windows,
            "top_20_videos_by_window_count": top_videos,
        }
