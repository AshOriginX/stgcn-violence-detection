"""
Object tracking module using ByteTrack.

This module provides production-quality object tracking with:
- Model warm-up for consistent inference timing
- Comprehensive error handling and logging
- FP16 support with CPU fallback
- Type hints and detailed docstrings
- Single responsibility principle
"""

import logging
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from pipeline.types import FrameResult, Track, RawTrack

logger = logging.getLogger(__name__)


class ByteTracker:
    """
    Production-quality ByteTrack-based object tracker.

    This class handles multi-object tracking using Ultralytics YOLO with ByteTrack,
    with support for warm-up, FP16 inference, and comprehensive error handling.

    Attributes
    ----------
    model : YOLO
        Loaded YOLO model instance with tracking capability.
    config : dict
        Tracker configuration dictionary.
    device : torch.device
        Computation device (CPU or CUDA).
    """

    def __init__(self, model: YOLO, config: dict):
        """
        Initialize the ByteTracker.

        Parameters
        ----------
        model : YOLO
            Loaded YOLO model instance with tracking capability.
        config : dict
            Tracker configuration with the following keys:
            - tracker: str, path to tracker configuration file (e.g., "bytetrack.yaml")
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

        logger.info(f"Initialized ByteTracker on device: {self.device}")

        if self.config.get("warmup", True):
            self._warmup()

    def _validate_config(self, config: dict) -> None:
        """
        Validate tracker configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If required configuration keys are missing.
        """

        required_keys = ["tracker", "imgsz"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def _resolve_device(self) -> torch.device:
        """
        Resolve the computation device based on model and system availability.

        Returns
        -------
        torch.device
            Appropriate computation device.
        """

        model_device = next(self.model.parameters()).device

        if model_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Falling back to CPU.")
            return torch.device("cpu")

        return model_device

    def _warmup(self) -> None:
        """
        Perform warm-up inference to initialize tracker and ensure consistent timing.

        This runs a few tracking passes with dummy data to:
        - Initialize GPU kernels
        - Allocate memory
        - Stabilize tracking timing
        - Initialize tracker state

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

        try:
            for i in range(warmup_frames):
                self.model.track(
                    dummy_frame,
                    tracker=self.config["tracker"],
                    persist=True,
                    verbose=False,
                )

            logger.info("Warm-up completed successfully.")

        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
            raise RuntimeError(f"Warm-up inference failed: {e}") from e

    def track(self, frame: np.ndarray, frame_result: FrameResult) -> FrameResult:
        """
        Run object tracking on a single frame.

        This method performs the following steps:
        1. Run YOLO tracking with ByteTrack on the frame
        2. Parse raw tracking results into structured Track objects
        3. Add tracks to the FrameResult
        4. Return the enriched FrameResult

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format with shape (H, W, 3).
        frame_result : FrameResult
            Frame result containing detections from the detector.

        Returns
        -------
        FrameResult
            Frame result with tracks added.

        Raises
        ------
        ValueError
            If frame is invalid or empty.
        RuntimeError
            If tracking fails.
        """

        if frame is None or frame.size == 0:
            raise ValueError(f"Invalid frame at index {frame_result.frame_index}")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be 3-channel BGR image, got shape {frame.shape}")

        try:
            results = self._run_tracking(frame)
            if not results:
                frame_result.tracks = []
                return frame_result
            raw_tracks = self._parse_results(results, frame.shape)
            tracks = self._create_track_objects(raw_tracks)

            frame_result.tracks = tracks

            logger.debug("Tracked %d objects at frame %d", len(tracks), frame_result.frame_index)

            return frame_result

        except Exception as e:
            logger.error(f"Tracking failed at frame {frame_result.frame_index}: {e}")
            raise RuntimeError(f"Tracking failed: {e}") from e

    def _run_tracking(self, frame: np.ndarray):
        """
        Run YOLO tracking with ByteTrack on a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format.

        Returns
        -------
        list
            YOLO tracking results.
        """

        return self.model.track(
            frame,
            tracker=self.config["tracker"],
            persist=True,
            verbose=False,
        )

    def _parse_results(self, results, frame_shape: tuple) -> List[RawTrack]:
        """
        Parse YOLO tracking result objects into raw tracking dictionaries.

        This method extracts raw data from Ultralytics Results objects
        without creating Track instances, maintaining single responsibility.

        Parameters
        ----------
        results : ultralytics.engine.results.Results or list
            YOLO tracking result object or list of result objects.
        frame_shape : tuple
            Frame shape (H, W) for clipping boxes to image boundaries.

        Returns
        -------
        List[RawTrack]
            List of raw tracking dictionaries with keys:
            - track_id: int, unique track identifier
            - bbox: List[float], bounding box in pixel coordinates [x1, y1, x2, y2]
            - confidence: float, detection confidence score
        """

        raw_tracks = []

        # Handle list or single Results object
        if isinstance(results, list):
            if not results:
                return raw_tracks
            results = results[0]

        if results.boxes is None:
            return raw_tracks

        for box in results.boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])
            bbox = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            # Clip boxes to image boundaries
            bbox = self._clip_bbox(bbox, frame_shape)

            raw_tracks.append(
                {"track_id": track_id, "bbox": bbox, "confidence": confidence}
            )

        return raw_tracks

    def _create_track_objects(self, raw_tracks: List[RawTrack]) -> List[Track]:
        """
        Create Track dataclass instances from raw tracking dictionaries.

        Parameters
        ----------
        raw_tracks : List[RawTrack]
            List of raw tracking dictionaries.

        Returns
        -------
        List[Track]
            List of Track objects sorted by track_id for deterministic ordering.
        """

        tracks = []

        for raw_track in raw_tracks:
            tracks.append(
                Track(
                    track_id=raw_track["track_id"],
                    bbox=raw_track["bbox"],
                    confidence=raw_track["confidence"],
                )
            )

        # Sort by track_id for deterministic ordering
        tracks.sort(key=lambda t: t.track_id)

        return tracks

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
            logger.info("ByteTracker resources released.")
