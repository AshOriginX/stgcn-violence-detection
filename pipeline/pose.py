"""
Pose estimation module using RTMPose.

This module provides production-quality pose estimation with:
- Model warm-up for consistent inference timing
- Comprehensive error handling and logging
- GPU support with CPU fallback
- Type hints and detailed docstrings
- Single responsibility principle
"""

import logging
from typing import List

import numpy as np
import torch
from mmpose.apis import inference_topdown

from pipeline.types import FrameResult, PoseResult, Track

logger = logging.getLogger(__name__)


class RTMPoseEstimator:
    """
    Production-quality RTMPose-based pose estimator.

    This class handles human pose estimation using MMPose RTMPose,
    with support for warm-up, GPU inference, and comprehensive error handling.

    Attributes
    ----------
    model : mmpose model
        Loaded RTMPose model instance.
    config : dict
        Pose estimator configuration dictionary.
    device : torch.device
        Computation device (CPU or CUDA).
    """

    def __init__(self, model, config: dict):
        """
        Initialize the RTMPose estimator.

        Parameters
        ----------
        model : mmpose model
            Loaded RTMPose model instance.
        config : dict
            Pose estimator configuration with the following keys:
            - device: str, computation device (e.g., "cuda:0", "cpu")
            - imgsz: int or tuple, input image size for warmup
            - warmup: bool, whether to perform warm-up inference
            - warmup_frames: int, number of warm-up frames

        Raises
        ------
        ValueError
            If required configuration keys are missing.
        """

        self._validate_config(config)

        self.model = model
        self.config = config
        self.device = self._resolve_device()

        logger.info(f"Initialized RTMPoseEstimator on device: {self.device}")

        if self.config.get("warmup", True):
            self._warmup()

    def _validate_config(self, config: dict) -> None:
        """
        Validate pose estimator configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If required configuration keys are missing.
        """

        required_keys = ["device", "imgsz"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def _resolve_device(self) -> torch.device:
        """
        Resolve the computation device based on configuration and system availability.

        Returns
        -------
        torch.device
            Appropriate computation device.
        """

        device_str = self.config["device"]

        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Falling back to CPU.")
            return torch.device("cpu")

        return torch.device(device_str)

    def _warmup(self) -> None:
        """
        Perform warm-up inference to initialize model and ensure consistent timing.

        This runs a few inference passes with dummy data to:
        - Initialize GPU kernels
        - Allocate memory
        - Stabilize inference timing

        Raises
        ------
        RuntimeError
            If warm-up inference fails.
        """

        warmup_frames = self.config.get("warmup_frames", 3)
        imgsz = self.config["imgsz"]
        
        # Handle imgsz as int or tuple
        if isinstance(imgsz, int):
            dummy_size = (imgsz, imgsz)
        else:
            dummy_size = imgsz
            
        logger.info("Performing warm-up with %d frames at size %s...", warmup_frames, dummy_size)

        dummy_frame = np.zeros((dummy_size[1], dummy_size[0], 3), dtype=np.uint8)
        dummy_bbox = [100, 100, 200, 200]

        try:
            for _ in range(warmup_frames):
                inference_topdown(self.model, dummy_frame, [dummy_bbox])

            logger.info("Warm-up completed successfully.")

        except Exception as e:
            logger.error("Warm-up failed: %s", e)
            raise RuntimeError(f"Warm-up inference failed: {e}") from e

    def estimate(self, frame: np.ndarray, frame_result: FrameResult) -> FrameResult:
        """
        Run pose estimation on a single frame.

        This method performs the following steps:
        1. Check if there are tracks to estimate poses for
        2. For each track, use bbox in pixel coordinates
        3. Run RTMPose inference on each track
        4. Parse raw pose results into structured PoseResult objects
        5. Add poses to the FrameResult
        6. Return the enriched FrameResult

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format with shape (H, W, 3).
        frame_result : FrameResult
            Frame result containing tracks from the tracker.

        Returns
        -------
        FrameResult
            Frame result with poses added.

        Raises
        ------
        ValueError
            If frame is invalid or empty.
        RuntimeError
            If pose estimation fails.
        """

        if frame is None or frame.size == 0:
            raise ValueError(f"Invalid frame at index {frame_result.frame_index}")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be 3-channel BGR image, got shape {frame.shape}")

        if not frame_result.tracks:
            logger.debug("No tracks at frame %d, skipping pose estimation", frame_result.frame_index)
            return frame_result

        try:
            poses = self._estimate_poses(frame, frame_result.tracks)
            frame_result.poses = poses

            logger.debug("Estimated %d poses at frame %d", len(poses), frame_result.frame_index)

            return frame_result

        except Exception as e:
            logger.error(f"Pose estimation failed at frame {frame_result.frame_index}: {e}")
            raise RuntimeError(f"Pose estimation failed: {e}") from e

    def _estimate_poses(self, frame: np.ndarray, tracks: List[Track]) -> List[PoseResult]:
        """
        Estimate poses for all tracked persons.

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format.
        tracks : List[Track]
            List of tracks with bounding boxes in pixel coordinates.

        Returns
        -------
        List[PoseResult]
            List of PoseResult objects sorted by track_id for deterministic ordering.
        """

        poses = []

        for track in tracks:
            # Bboxes are already in xyxy pixel coordinates from tracker
            bbox = self._clip_bbox(track.bbox, frame.shape)
            raw_pose = self._run_inference(frame, bbox)

            if raw_pose is None:
                continue

            pose_result = self._create_pose_result(track.track_id, raw_pose)
            if pose_result is not None:
                poses.append(pose_result)

        # Sort by track_id for deterministic ordering
        poses.sort(key=lambda p: p.track_id)

        return poses

    def _run_inference(self, frame: np.ndarray, bbox: List[float]):
        """
        Run RTMPose inference on a single bounding box.

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format.
        bbox : List[float]
            Bounding box in pixel coordinates [x1, y1, x2, y2].

        Returns
        -------
        mmpose result or None
            Raw pose estimation result, or None if inference fails.
        """

        try:
            pose_result = inference_topdown(self.model, frame, [bbox])

            if pose_result[0].pred_instances is None:
                return None

            return pose_result[0]

        except Exception as e:
            logger.warning(f"Pose inference failed for bbox {bbox}: {e}")
            return None

    def _create_pose_result(self, track_id: int, raw_pose) -> PoseResult:
        """
        Create a PoseResult object from raw pose estimation data.

        Parameters
        ----------
        track_id : int
            Track identifier.
        raw_pose : mmpose result
            Raw pose estimation result.

        Returns
        -------
        PoseResult
            Structured pose result object, or None if validation fails.
        """

        keypoints = raw_pose.pred_instances.keypoints[0].tolist()
        scores = raw_pose.pred_instances.keypoint_scores[0].tolist()

        # Validate keypoint count (RTMPose typically has 17 keypoints for COCO format)
        expected_keypoints = 17
        if len(keypoints) != expected_keypoints:
            logger.warning("Pose for track %d has %d keypoints, expected %d. Skipping.", track_id, len(keypoints), expected_keypoints)
            return None

        return PoseResult(track_id=track_id, keypoints=keypoints, scores=scores)

    def _clip_bbox(self, bbox: List[float], frame_shape: tuple) -> List[float]:
        """
        Clip bounding box coordinates to image boundaries.

        Parameters
        ----------
        bbox : List[float]
            Bounding box in pixel coordinates [x1, y1, x2, y2].
        frame_shape : tuple
            Frame shape (H, W).

        Returns
        -------
        List[float]
            Clipped bounding box.
        """

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        return [x1, y1, x2, y2]

    def close(self) -> None:
        """
        Clean up resources.

        This method releases model resources and performs cleanup.
        """

        if hasattr(self, "model"):
            del self.model
            logger.info("RTMPoseEstimator resources released.")
