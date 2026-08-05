"""
Pipeline extractor module for orchestrating the preprocessing pipeline.

This module provides production-quality pipeline orchestration with:
- Complete pipeline coordination (detector, tracker, pose, exporter)
- Comprehensive error handling and logging
- Progress tracking and reporting
- Resource management and cleanup
- Type hints and detailed docstrings
- Single responsibility principle
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pipeline.detector import YOLODetector
from pipeline.pose import RTMPoseEstimator
from pipeline.tracker import ByteTracker
from pipeline.types import FrameResult, VideoResult

logger = logging.getLogger(__name__)


class PipelineExtractor:
    """
    Production-quality pipeline orchestrator for video preprocessing.

    This class coordinates the entire preprocessing pipeline:
    1. Object detection using YOLO
    2. Multi-object tracking using ByteTrack
    3. Pose estimation using RTMPose

    Attributes
    ----------
    detector : YOLODetector
        Object detection instance.
    tracker : ByteTracker
        Object tracking instance.
    pose_estimator : RTMPoseEstimator
        Pose estimation instance.
    config : dict
        Pipeline configuration.
    """

    def __init__(
        self,
        detector: YOLODetector,
        tracker: ByteTracker,
        pose_estimator: RTMPoseEstimator,
        config: dict,
    ):
        """
        Initialize the pipeline extractor.

        Parameters
        ----------
        detector : YOLODetector
            Initialized YOLO detector instance.
        tracker : ByteTracker
            Initialized ByteTracker instance.
        pose_estimator : RTMPoseEstimator
            Initialized RTMPose estimator instance.
        config : dict
            Pipeline configuration with the following keys:
            - skip_frames: int, number of frames to skip between processing
            - max_frames: int, maximum number of frames to process (None for all)

        Raises
        ------
        ValueError
            If required configuration keys are missing.
        """

        self._validate_config(config)

        self.detector = detector
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.config = config

        logger.info("Initialized PipelineExtractor with all components")

    def _validate_config(self, config: dict) -> None:
        """
        Validate pipeline configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If required configuration keys are missing.
        """

        required_keys = ["skip_frames", "max_frames"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def process_video(self, video_path: Path, video_id: str, label: int) -> VideoResult:
        """
        Process a complete video through the pipeline.

        This method:
        1. Opens the video file
        2. Iterates through frames with optional skipping
        3. Runs detection, tracking, and pose estimation
        4. Returns VideoResult with all frame results

        Parameters
        ----------
        video_path : Path
            Path to the video file.
        video_id : str
            Video identifier.
        label : int
            Action label for the video.

        Returns
        -------
        VideoResult
            Video result containing all frame results.

        Raises
        ------
        ValueError
            If video path is invalid or video cannot be opened.
        RuntimeError
            If pipeline processing fails.
        """

        if video_path is None or not video_path.exists():
            raise ValueError(f"Invalid video path: {video_path}")

        logger.info("Processing video: %s (ID: %s, Label: %d)", video_path, video_id, label)

        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_results = self._process_frames(cap, video_id, label)

            cap.release()

            video_result = VideoResult(
                video_id=video_id,
                label=label,
                width=frame_width,
                height=frame_height,
                total_frames=len(frame_results),
                frames=frame_results,
            )

            logger.info("Completed processing video %s: %d frames processed", video_id, len(frame_results))

            return video_result

        except Exception as e:
            logger.error("Failed to process video %s: %s", video_path, e)
            raise RuntimeError(f"Video processing failed: {e}")

    def _process_frames(self, cap: cv2.VideoCapture, video_id: str, label: int) -> List[FrameResult]:
        """
        Process all frames from a video capture.

        Parameters
        ----------
        cap : cv2.VideoCapture
            OpenCV video capture object.
        video_id : str
            Video identifier.
        label : int
            Action label.

        Returns
        -------
        List[FrameResult]
            List of processed frame results.
        """

        frame_results = []
        frame_index = 0
        processed_count = 0

        skip_frames = self.config["skip_frames"]
        max_frames = self.config.get("max_frames")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {total_frames}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if max_frames is not None and processed_count >= max_frames:
                logger.info(f"Reached maximum frame limit: {max_frames}")
                break

            if frame_index % (skip_frames + 1) != 0:
                frame_index += 1
                continue

            frame_result = self._process_single_frame(frame, frame_index, video_id)

            frame_results.append(frame_result)

            processed_count += 1
            frame_index += 1

            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{total_frames} frames")

        logger.info(f"Processed {processed_count} frames from video")

        return frame_results

    def _process_single_frame(self, frame: np.ndarray, frame_index: int, video_id: str) -> FrameResult:
        """
        Process a single frame through the complete pipeline.

        This method orchestrates:
        1. Object detection
        2. Object tracking
        3. Pose estimation

        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format.
        frame_index : int
            Frame index in the video.
        video_id : str
            Video identifier for logging.

        Returns
        -------
        FrameResult
            Frame result with detections, tracks, and poses.

        Raises
        ------
        RuntimeError
            If any pipeline stage fails.
        """

        try:
            frame_result = self.detector.detect(frame, frame_index)
            frame_result = self.tracker.track(frame, frame_result)
            frame_result = self.pose_estimator.estimate(frame, frame_result)

            logger.debug(f"Processed frame {frame_index}: {len(frame_result.detections)} detections, "
                        f"{len(frame_result.tracks)} tracks, {len(frame_result.poses)} poses")

            return frame_result

        except Exception as e:
            logger.error(f"Failed to process frame {frame_index} in video {video_id}: {e}")
            raise RuntimeError(f"Frame processing failed: {e}")

    def process_batch(self, video_paths: List[Path], video_ids: List[str], labels: List[int]) -> dict:
        """
        Process a batch of videos through the pipeline.

        Parameters
        ----------
        video_paths : List[Path]
            List of video file paths.
        video_ids : List[str]
            List of video identifiers.
        labels : List[int]
            List of action labels.

        Returns
        -------
        dict
            Dictionary mapping video IDs to VideoResult objects.

        Raises
        ------
        ValueError
            If input lists have mismatched lengths.
        """

        if len(video_paths) != len(video_ids) or len(video_paths) != len(labels):
            raise ValueError("video_paths, video_ids, and labels must have the same length")

        logger.info("Processing batch of %d videos", len(video_paths))

        results = {}

        for i, (video_path, video_id, label) in enumerate(zip(video_paths, video_ids, labels)):
            logger.info("Processing video %d/%d: %s", i+1, len(video_paths), video_id)

            try:
                video_result = self.process_video(video_path, video_id, label)
                results[video_id] = video_result

            except Exception as e:
                logger.error("Failed to process video %s: %s", video_id, e)
                results[video_id] = None

        successful = sum(1 for r in results.values() if r is not None)
        logger.info("Batch processing complete: %d/%d videos successful", successful, len(video_paths))

        return results

    def close(self) -> None:
        """
        Clean up all pipeline resources.

        This method releases resources from all pipeline components.
        """

        logger.info("Closing pipeline resources...")

        if hasattr(self, "detector"):
            self.detector.close()

        if hasattr(self, "tracker"):
            self.tracker.close()

        if hasattr(self, "pose_estimator"):
            self.pose_estimator.close()

        logger.info("Pipeline resources released")
