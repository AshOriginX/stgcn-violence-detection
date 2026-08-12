"""
Reusable preprocessing functions for ST-GCN++ inference and training.

This module extracts preprocessing logic from STGCNDataset to enable
direct inference on raw keypoint arrays without PKL file dependency.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WindowIndex:
    """Index information for a temporal window."""
    annotation_idx: int
    window_start: int
    window_end: int
    video_id: str
    label: int


def handle_invalid_values(
    keypoint: np.ndarray,
    invalid_value_policy: str = "zero"
) -> np.ndarray:
    """
    Handle NaN and Inf values in keypoint array.

    Parameters
    ----------
    keypoint : np.ndarray
        Keypoint array to process.
    invalid_value_policy : str
        Policy for handling invalid values: "error" or "zero".

    Returns
    -------
    np.ndarray
        Keypoint array with invalid values handled.

    Raises
    ------
    ValueError
        If invalid_value_policy is "error" and NaN/Inf are found.
    """
    has_nan = np.isnan(keypoint).any()
    has_inf = np.isinf(keypoint).any()

    if has_nan or has_inf:
        if invalid_value_policy == "error":
            raise ValueError(
                f"Keypoint contains {'NaN' if has_nan else ''}{'Inf' if has_inf else ''} values"
            )
        else:  # "zero" policy
            keypoint = np.nan_to_num(keypoint, nan=0.0, posinf=0.0, neginf=0.0)

    return keypoint


def normalize_persons(keypoint: np.ndarray, target_m: int = 2) -> np.ndarray:
    """
    Normalize person dimension to exactly target_m.

    Parameters
    ----------
    keypoint : np.ndarray
        Keypoint array with shape (M, T, V, C).
    target_m : int
        Target number of persons.

    Returns
    -------
    np.ndarray
        Keypoint array with shape (target_m, T, V, C).
    """
    M, T, V, C = keypoint.shape

    if M == target_m:
        return keypoint

    if M < target_m:
        # Pad with zeros
        padding = np.zeros((target_m - M, T, V, C), dtype=keypoint.dtype)
        return np.concatenate([keypoint, padding], axis=0)

    # M > target_m should have been caught in validation
    raise ValueError(f"Unexpected person dimension: {M} > target {target_m}")


def temporal_resample(
    keypoint: np.ndarray,
    clip_len: int
) -> np.ndarray:
    """
    Normalize temporal dimension to exactly clip_len frames.

    With windowing enabled, this only handles padding for short videos.
    Long videos are already windowed to clip_len before this is called.

    Parameters
    ----------
    keypoint : np.ndarray
        Keypoint array with shape (M, T, V, C).
    clip_len : int
        Target temporal length.

    Returns
    -------
    np.ndarray
        Keypoint array with shape (M, clip_len, V, C).
    """
    M, T, V, C = keypoint.shape

    if T == clip_len:
        return keypoint

    if T > clip_len:
        # This should not happen with windowing enabled
        # Fall back to uniform sampling
        indices = np.linspace(0, T - 1, clip_len).round().astype(np.int64)
        return keypoint[:, indices, :, :]

    # T < clip_len: pad by repeating the last frame
    pad_len = clip_len - T
    last_frame = keypoint[:, -1:, :, :]  # (M, 1, V, C)
    padding = np.repeat(last_frame, pad_len, axis=1)  # (M, pad_len, V, C)
    return np.concatenate([keypoint, padding], axis=1)  # (M, clip_len, V, C)


def spatial_normalize(keypoint: np.ndarray) -> np.ndarray:
    """
    Apply per-window spatial normalization (z-score standardization).

    For each temporal window independently:
    - Collect valid (x,y) coordinates where coordinate is not (0,0)
    - Compute separate X/Y statistics: mean = np.mean(valid_coords, axis=0)
    - Compute separate X/Y statistics: std = np.std(valid_coords, axis=0)
    - Clamp very small standard deviations: std[std < 1e-6] = 1.0
    - Normalize valid coordinates: normalized = (coord - mean) / std
    - (0,0) missing/invalid joints remain exactly (0,0)
    - Zero-padded persons remain entirely zero
    - Both persons use the SAME window-level mean/std

    Parameters
    ----------
    keypoint : np.ndarray
        Keypoint array with shape (M, T, V, C).

    Returns
    -------
    np.ndarray
        Spatially normalized keypoint array with shape (M, T, V, C).
    """
    M, T, V, C = keypoint.shape
    normalized = keypoint.copy()

    # Extract (x,y) coordinates: shape (M, T, V, 2)
    coords = keypoint[:, :, :, :2]

    # Create mask for valid coordinates: True if (x,y) != (0,0)
    # Shape: (M, T, V)
    valid_mask = ~np.all(coords == 0, axis=-1)

    if valid_mask.any():
        # Extract valid coordinates for statistics computation
        # Shape: (N, 2) where N is number of valid coordinates
        valid_coords = coords[valid_mask]

        # Compute separate X/Y statistics
        mean = np.mean(valid_coords, axis=0)  # Shape: (2,)
        std = np.std(valid_coords, axis=0)    # Shape: (2,)

        # Clamp very small standard deviations
        std[std < 1e-6] = 1.0

        # Normalize only valid coordinates using vectorized operations
        # Expand mean/std to broadcast: (1, 1, 1, 2)
        mean_expanded = mean.reshape(1, 1, 1, 2)
        std_expanded = std.reshape(1, 1, 1, 2)

        # Apply normalization to all coordinates
        normalized_coords = (coords - mean_expanded) / std_expanded

        # Only update valid coordinates, preserve zeros
        # Expand valid_mask to (M, T, V, 1) for broadcasting
        valid_mask_expanded = valid_mask[:, :, :, np.newaxis]
        normalized[:, :, :, :2] = np.where(
            valid_mask_expanded,
            normalized_coords,
            coords  # Keep original (0,0) for invalid coordinates
        )

    return normalized


def generate_windows(
    keypoint: np.ndarray,
    video_id: str,
    label: int,
    clip_len: int,
    window_stride: int,
    enable_windowing: bool,
    max_windows_per_video: int = None
) -> List[WindowIndex]:
    """
    Generate temporal windows from a keypoint array.

    Parameters
    ----------
    keypoint : np.ndarray
        Keypoint array with shape (M, T, V, C).
    video_id : str
        Video identifier.
    label : int
        Video label.
    clip_len : int
        Target window length.
    window_stride : int
        Stride for window extraction.
    enable_windowing : bool
        Whether to enable sliding window extraction.
    max_windows_per_video : int, optional
        Maximum number of windows per video.

    Returns
    -------
    List[WindowIndex]
        List of window indices.
    """
    M, T, V, C = keypoint.shape
    video_windows = []

    if T == clip_len:
        # Exactly clip_len: one window
        video_windows.append(
            WindowIndex(
                annotation_idx=0,
                window_start=0,
                window_end=T,
                video_id=video_id,
                label=label,
            )
        )
    elif T < clip_len:
        # Short video: one window (will be padded)
        video_windows.append(
            WindowIndex(
                annotation_idx=0,
                window_start=0,
                window_end=T,
                video_id=video_id,
                label=label,
            )
        )
    else:  # T > clip_len
        if enable_windowing:
            # Extract overlapping windows
            for start in range(0, T - clip_len + 1, window_stride):
                end = min(start + clip_len, T)
                # Ensure last window is exactly clip_len
                if end - start < clip_len:
                    start = T - clip_len
                    end = T
                video_windows.append(
                    WindowIndex(
                        annotation_idx=0,
                        window_start=start,
                        window_end=end,
                        video_id=video_id,
                        label=label,
                    )
                )
        else:
            # Old behavior: uniformly sample entire video
            video_windows.append(
                WindowIndex(
                    annotation_idx=0,
                    window_start=0,
                    window_end=T,
                    video_id=video_id,
                    label=label,
                )
            )

    # Apply cap if configured
    if max_windows_per_video is not None and len(video_windows) > max_windows_per_video:
        # Deterministically select windows uniformly across temporal range
        indices = np.linspace(0, len(video_windows) - 1, max_windows_per_video).round().astype(int)
        # Ensure uniqueness
        indices = np.unique(indices)
        # Sort to preserve temporal ordering
        indices.sort()
        video_windows = [video_windows[i] for i in indices]

    return video_windows


def extract_window(
    keypoint: np.ndarray,
    window: WindowIndex,
    clip_len: int
) -> np.ndarray:
    """
    Extract a temporal window from keypoint array.

    Parameters
    ----------
    keypoint : np.ndarray
        Full keypoint array with shape (M, T, V, C).
    window : WindowIndex
        Window index information.
    clip_len : int
        Target window length.

    Returns
    -------
    np.ndarray
        Windowed keypoint array with shape (M, clip_len, V, C).
    """
    window_keypoint = keypoint[:, window.window_start:window.window_end, :, :]
    return temporal_resample(window_keypoint, clip_len)


def preprocess_window(
    keypoint: np.ndarray,
    clip_len: int,
    enable_normalization: bool,
    invalid_value_policy: str = "zero"
) -> np.ndarray:
    """
    Apply complete preprocessing pipeline to a single window.

    Parameters
    ----------
    keypoint : np.ndarray
        Input keypoint array with shape (M, T, V, C).
    clip_len : int
        Target temporal length.
    enable_normalization : bool
        Whether to apply spatial normalization.
    invalid_value_policy : str
        Policy for handling invalid values.

    Returns
    -------
    np.ndarray
        Preprocessed keypoint array with shape (2, clip_len, 17, 2).
    """
    # Handle invalid values
    keypoint = handle_invalid_values(keypoint, invalid_value_policy)

    # Normalize temporal dimension
    keypoint = temporal_resample(keypoint, clip_len)

    # Normalize person dimension to exactly 2
    keypoint = normalize_persons(keypoint, target_m=2)

    # Apply spatial normalization
    if enable_normalization:
        keypoint = spatial_normalize(keypoint)

    return keypoint
