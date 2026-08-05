"""
Export module for transforming pipeline results to PySKL annotation format.

This module provides production-quality export functionality with:
- PySKL-compatible annotation dictionary generation
- Track-based temporal structure preservation
- Comprehensive error handling and logging
- Type hints and detailed docstrings
- Single responsibility principle
"""

import logging
from typing import Dict, List

import numpy as np

from pipeline.types import PoseResult, VideoResult

logger = logging.getLogger(__name__)


class PipelineExporter:
    """
    Production-quality exporter for PySKL annotation format.

    This class transforms VideoResult objects into PySKL-compatible annotation
    dictionaries, preserving temporal structure and grouping poses by track_id.

    The exporter does NOT write files - it only performs data transformation.
    File writing is handled by dataset_builder.py.

    Attributes
    ----------
    max_tracks : int
        Maximum number of tracks to include in the annotation.
    num_keypoints : int
        Expected number of keypoints per pose (17 for COCO format).
    """

    def __init__(self, max_tracks: int = 2, num_keypoints: int = 17):
        """
        Initialize the pipeline exporter.

        Parameters
        ----------
        max_tracks : int, optional
            Maximum number of tracks to include in the annotation. Default is 2.
        num_keypoints : int, optional
            Expected number of keypoints per pose. Default is 17 for COCO format.

        Raises
        ------
        ValueError
            If max_tracks or num_keypoints are invalid.
        """

        if max_tracks <= 0:
            raise ValueError("max_tracks must be positive")

        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")

        self.max_tracks = max_tracks
        self.num_keypoints = num_keypoints

        logger.info(
            "Initialized PipelineExporter with max_tracks=%d, num_keypoints=%d",
            max_tracks,
            num_keypoints,
        )

    def export(self, video_result: VideoResult) -> Dict:
        """
        Transform a VideoResult into a PySKL-compatible annotation dictionary.

        This method performs the following steps:
        1. Group poses by track_id to preserve temporal structure
        2. Select top tracks by pose count
        3. Build keypoint tensor (M, T, K, 2)
        4. Build score tensor (M, T, K)
        5. Create annotation dictionary
        6. Validate annotation

        Parameters
        ----------
        video_result : VideoResult
            Video result containing frame-level pose data.

        Returns
        -------
        Dict
            PySKL-compatible annotation dictionary with keys:
            - frame_dir: str, video identifier
            - label: int, action label
            - img_shape: tuple, (height, width)
            - original_shape: tuple, (height, width)
            - total_frames: int, total number of frames
            - keypoint: np.ndarray, shape (M, T, K, 2)
            - keypoint_score: np.ndarray, shape (M, T, K)

        Raises
        ------
        ValueError
            If video_result is invalid or has no poses.
        RuntimeError
            If annotation transformation fails.
        """

        if video_result is None:
            raise ValueError("video_result cannot be None")

        if not video_result.frames:
            raise ValueError("video_result must contain at least one frame")

        try:
            # Group poses by track_id
            track_poses = self._group_tracks(video_result.frames)

            if not track_poses:
                raise ValueError("No poses found in video_result")

            logger.debug(
                "Grouped %d tracks from %d frames",
                len(track_poses),
                len(video_result.frames),
            )

            # Select top tracks
            selected_tracks = self._select_top_tracks(track_poses)

            logger.debug("Selected top %d tracks", len(selected_tracks))

            # Build tensors
            keypoints = self._build_keypoint_tensor(
                selected_tracks, video_result.total_frames
            )
            scores = self._build_score_tensor(
                selected_tracks, video_result.total_frames
            )

            # Create annotation
            annotation = self._create_annotation(
                video_result, keypoints, scores
            )

            # Validate annotation
            self._validate_annotation(annotation)

            logger.info(
                "Created annotation for video %s with shape %s",
                video_result.video_id,
                annotation["keypoint"].shape,
            )

            return annotation

        except Exception as e:
            logger.error("Failed to export video %s: %s", video_result.video_id, e)
            raise RuntimeError(f"Export failed: {e}") from e

    def _group_tracks(self, frames: List) -> Dict[int, Dict[int, PoseResult]]:
        """
        Group poses by track_id across all frames.

        This method creates a nested structure:
        {
            track_id: {
                frame_index: PoseResult,
                ...
            },
            ...
        }

        Parameters
        ----------
        frames : List
            List of FrameResult objects.

        Returns
        -------
        Dict[int, Dict[int, PoseResult]]
            Nested dictionary mapping track_id to frame_index to PoseResult.
        """

        track_poses = {}

        for frame in frames:
            for pose in frame.poses:
                if pose.track_id not in track_poses:
                    track_poses[pose.track_id] = {}

                track_poses[pose.track_id][frame.frame_index] = pose

        return track_poses

    def _select_top_tracks(
        self, track_poses: Dict[int, Dict[int, PoseResult]]
    ) -> List[tuple]:
        """
        Select top tracks by pose count.

        Parameters
        ----------
        track_poses : Dict[int, Dict[int, PoseResult]]
            Nested dictionary of track poses.

        Returns
        -------
        List[tuple]
            Top N tracks as (track_id, frame_poses) tuples sorted by pose count.
        """

        # Sort tracks by pose count (descending)
        sorted_tracks = sorted(
            track_poses.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        # Select top N tracks
        selected = sorted_tracks[: self.max_tracks]

        return selected

    def _build_keypoint_tensor(
        self,
        track_poses: List[tuple],
        total_frames: int,
    ) -> np.ndarray:
        """
        Build keypoint tensor with shape (M, T, K, 2).

        Parameters
        ----------
        track_poses : List[tuple]
            List of (track_id, frame_poses) tuples.
        total_frames : int
            Total number of frames in the video.

        Returns
        -------
        np.ndarray
            Keypoint tensor with shape (M, T, K, 2), where:
            - M: number of tracks (max_tracks)
            - T: total frames
            - K: number of keypoints (num_keypoints)
            - 2: x, y coordinates

        Missing poses are filled with zeros.
        """

        M = self.max_tracks
        T = total_frames
        K = self.num_keypoints

        keypoints = np.zeros((M, T, K, 2), dtype=np.float32)

        # Fill in keypoints for each track
        for track_idx, (track_id, frame_poses) in enumerate(track_poses):
            for frame_idx, pose in frame_poses.items():
                if frame_idx < T:
                    pose_keypoints = np.array(pose.keypoints, dtype=np.float32)

                    if len(pose_keypoints) == K:
                        keypoints[track_idx, frame_idx] = pose_keypoints
                    else:
                        logger.warning(
                            "Track %d frame %d has %d keypoints, expected %d",
                            track_id,
                            frame_idx,
                            len(pose_keypoints),
                            K,
                        )

        return keypoints

    def _build_score_tensor(
        self,
        track_poses: List[tuple],
        total_frames: int,
    ) -> np.ndarray:
        """
        Build keypoint score tensor with shape (M, T, K).

        Parameters
        ----------
        track_poses : List[tuple]
            List of (track_id, frame_poses) tuples.
        total_frames : int
            Total number of frames in the video.

        Returns
        -------
        np.ndarray
            Score tensor with shape (M, T, K), where:
            - M: number of tracks (max_tracks)
            - T: total frames
            - K: number of keypoints (num_keypoints)

        Missing poses are filled with zeros.
        """

        M = self.max_tracks
        T = total_frames
        K = self.num_keypoints

        scores = np.zeros((M, T, K), dtype=np.float32)

        # Fill in scores for each track
        for track_idx, (track_id, frame_poses) in enumerate(track_poses):
            for frame_idx, pose in frame_poses.items():
                if frame_idx < T:
                    pose_scores = np.array(pose.scores, dtype=np.float32)

                    if len(pose_scores) == K:
                        scores[track_idx, frame_idx] = pose_scores
                    else:
                        logger.warning(
                            "Track %d frame %d has %d scores, expected %d",
                            track_id,
                            frame_idx,
                            len(pose_scores),
                            K,
                        )

        return scores

    def _create_annotation(
        self,
        video_result: VideoResult,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> Dict:
        """
        Create PySKL-compatible annotation dictionary.

        Parameters
        ----------
        video_result : VideoResult
            Video result containing metadata.
        keypoints : np.ndarray
            Keypoint tensor with shape (M, T, K, 2).
        scores : np.ndarray
            Score tensor with shape (M, T, K).

        Returns
        -------
        Dict
            PySKL-compatible annotation dictionary.
        """

        return {
            "frame_dir": video_result.video_id,
            "label": video_result.label,
            "img_shape": (video_result.height, video_result.width),
            "original_shape": (video_result.height, video_result.width),
            "total_frames": video_result.total_frames,
            "keypoint": keypoints,
            "keypoint_score": scores,
        }

    def _validate_annotation(self, annotation: Dict) -> None:
        """
        Validate annotation dictionary structure and values.

        Parameters
        ----------
        annotation : Dict
            Annotation dictionary to validate.

        Raises
        ------
        ValueError
            If annotation is invalid.
        """

        required_keys = [
            "frame_dir",
            "label",
            "img_shape",
            "original_shape",
            "total_frames",
            "keypoint",
            "keypoint_score",
        ]

        for key in required_keys:
            if key not in annotation:
                raise ValueError(f"Missing required key: {key}")

        # Validate shapes
        keypoints = annotation["keypoint"]
        scores = annotation["keypoint_score"]

        if len(keypoints.shape) != 4:
            raise ValueError(
                f"keypoint must have 4 dimensions, got {len(keypoints.shape)}"
            )

        if len(scores.shape) != 3:
            raise ValueError(
                f"keypoint_score must have 3 dimensions, got {len(scores.shape)}"
            )

        if keypoints.shape[0] != scores.shape[0]:
            raise ValueError(
                f"keypoint and keypoint_score must have same first dimension, "
                f"got {keypoints.shape[0]} and {scores.shape[0]}"
            )

        if keypoints.shape[1] != scores.shape[1]:
            raise ValueError(
                f"keypoint and keypoint_score must have same second dimension, "
                f"got {keypoints.shape[1]} and {scores.shape[1]}"
            )

        if keypoints.shape[2] != scores.shape[2]:
            raise ValueError(
                f"keypoint and keypoint_score must have same third dimension, "
                f"got {keypoints.shape[2]} and {scores.shape[2]}"
            )

        if keypoints.shape[2] != self.num_keypoints:
            raise ValueError(
                f"keypoint must have {self.num_keypoints} keypoints, "
                f"got {keypoints.shape[2]}"
            )

        # Validate total_frames matches tensor dimension
        if annotation["total_frames"] != keypoints.shape[1]:
            raise ValueError(
                f"total_frames ({annotation['total_frames']}) must match "
                f"keypoint tensor second dimension ({keypoints.shape[1]})"
            )

        # Check for NaN values
        if np.isnan(keypoints).any():
            raise ValueError("keypoint tensor contains NaN values")

        if np.isnan(scores).any():
            raise ValueError("keypoint_score tensor contains NaN values")

        logger.debug("Annotation validation passed")

    def close(self) -> None:
        """
        Clean up resources.

        This method performs any necessary cleanup operations.
        """

        logger.info("PipelineExporter resources released.")
