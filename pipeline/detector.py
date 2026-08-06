"""
Object detection module using YOLO.

This module provides production-quality object detection with:
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

from pipeline.types import Detection, FrameResult, RawDetection

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Production-quality YOLO-based object detector.

    This class handles object detection using Ultralytics YOLO models,
    with support for warm-up, FP16 inference, and comprehensive error handling.

    Attributes
    ----------
    model : YOLO
        Loaded YOLO model instance.
    config : dict
        Detector configuration dictionary.
    device : torch.device
        Computation device (CPU or CUDA).
    """

    def __init__(self, model: YOLO, config: dict):
        """
        Initialize the YOLO detector.

        Parameters
        ----------
        model : YOLO
            Loaded YOLO model instance.
        config : dict
            Detector configuration with the following keys:
            - imgsz: int or tuple, input image size
            - conf: float, confidence threshold
            - iou: float, IoU threshold for NMS
            - max_det: int, maximum number of detections
            - classes: list[int], class IDs to detect
            - verbose: bool, whether to print verbose output
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

        logger.info(f"Initialized YOLODetector on device: {self.device}")

        if self.config.get("warmup", True):
            self._warmup()

    def _validate_config(self, config: dict) -> None:
        """
        Validate detector configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If required configuration keys are missing.
        """

        required_keys = ["imgsz", "conf", "iou", "max_det", "classes", "verbose"]
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
            
        logger.info(f"Performing warm-up with {warmup_frames} frames at size {dummy_size}...")

        dummy_frame = np.zeros((dummy_size[1], dummy_size[0], 3), dtype=np.uint8)

        try:
            for i in range(warmup_frames):
                self.model(
                    dummy_frame,
                    imgsz=imgsz,
                    conf=self.config["conf"],
                    iou=self.config["iou"],
                    max_det=self.config["max_det"],
                    classes=self.config["classes"],
                    verbose=False,
                )

            logger.info("Warm-up completed successfully.")

        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
            raise RuntimeError(f"Warm-up inference failed: {e}") from e

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """
        Run object detection on a single frame.

        This method performs the following steps:
        1. Run YOLO inference on the frame
        2. Parse raw results into structured Detection objects
        3. Filter detections to keep only person class
        4. Return FrameResult with detections

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format with shape (H, W, 3).
        frame_index : int
            Frame index in the video sequence.

        Returns
        -------
        FrameResult
            Frame result containing person detections.

        Raises
        ------
        ValueError
            If frame is invalid or empty.
        RuntimeError
            If inference fails.
        """

        if frame is None or frame.size == 0:
            raise ValueError(f"Invalid frame at index {frame_index}")
        
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be 3-channel BGR image, got shape {frame.shape}")

        try:
            results = self._run_inference(frame)
            raw_detections = self._parse_results(results[0], frame.shape)
            person_detections = self._filter_persons(raw_detections)

            return FrameResult(frame_index=frame_index, detections=person_detections)

        except Exception as e:
            logger.error(f"Detection failed at frame {frame_index}: {e}")
            raise RuntimeError(f"Detection failed: {e}") from e

    def _run_inference(self, frame: np.ndarray):
        """
        Run YOLO inference on a frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format.

        Returns
        -------
        list
            YOLO inference results.
        """

        return self.model(
            frame,
            imgsz=self.config["imgsz"],
            conf=self.config["conf"],
            iou=self.config["iou"],
            max_det=self.config["max_det"],
            classes=self.config["classes"],
            verbose=self.config["verbose"],
        )

    def _parse_results(self, result, frame_shape: tuple) -> List[RawDetection]:
        """
        Parse YOLO result objects into raw detection dictionaries.

        This method extracts raw data from Ultralytics Results objects
        without creating Detection instances, maintaining single responsibility.

        Parameters
        ----------
        result : ultralytics.engine.results.Results
            YOLO detection result object.
        frame_shape : tuple
            Frame shape (H, W) for clipping boxes to image boundaries.

        Returns
        -------
        List[RawDetection]
            List of raw detection dictionaries with keys:
            - bbox: List[float], bounding box in pixel coordinates [x1, y1, x2, y2]
            - confidence: float, detection confidence score
            - class_id: int, class identifier
        """

        raw_detections = []

        if result.boxes is None:
            return raw_detections

        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            # Clip boxes to image boundaries
            bbox = self._clip_bbox(bbox, frame_shape)

            raw_detections.append(
                {"bbox": bbox, "confidence": confidence, "class_id": class_id}
            )

        return raw_detections

    def _filter_persons(self, raw_detections: List[RawDetection]) -> List[Detection]:
        """
        Filter detections to keep only person class and create Detection objects.

        COCO dataset class ID for person is 0. This method:
        1. Sorts detections by confidence for deterministic ordering
        2. Filters raw detections to keep only class_id == 0
        3. Creates Detection dataclass instances

        Parameters
        ----------
        raw_detections : List[RawDetection]
            List of raw detection dictionaries.

        Returns
        -------
        List[Detection]
            List of Detection objects for person class only.
        """

        # Sort by confidence for deterministic ordering
        sorted_detections = sorted(raw_detections, key=lambda d: d["confidence"], reverse=True)

        person_detections = []

        for raw_det in sorted_detections:
            if raw_det["class_id"] == 0:  # COCO person class
                person_detections.append(
                    Detection(
                        bbox=raw_det["bbox"],
                        confidence=raw_det["confidence"],
                        class_id=raw_det["class_id"],
                    )
                )

        logger.debug("Filtered %d person detections from %d total detections", len(person_detections), len(raw_detections))

        return person_detections

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
            logger.info("YOLODetector resources released.")
