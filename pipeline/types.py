"""
Common data structures for the ST-GCN++ preprocessing pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict


# ==========================================================
# Raw Detection Dictionary
# ==========================================================

class RawDetection(TypedDict):
    """Raw detection dictionary from YOLO."""
    bbox: List[float]
    confidence: float
    class_id: int


# ==========================================================
# Raw Track Dictionary
# ==========================================================

class RawTrack(TypedDict):
    """Raw tracking dictionary from ByteTrack."""
    track_id: int
    bbox: List[float]
    confidence: float


# ==========================================================
# Detection
# ==========================================================

@dataclass
class Detection:
    """
    Single object detection.
    """

    bbox: List[float]
    confidence: float
    class_id: int


# ==========================================================
# Track
# ==========================================================

@dataclass
class Track:
    """
    Single tracked object.
    """

    track_id: int
    bbox: List[float]
    confidence: float


# ==========================================================
# Pose
# ==========================================================

@dataclass
class PoseResult:
    """
    Pose estimation result for one tracked person.
    """

    track_id: int

    keypoints: List[List[float]]

    scores: List[float]


# ==========================================================
# Skeleton Sequence
# ==========================================================

@dataclass
class SkeletonSequence:
    """
    Skeleton sequence for one video.
    """

    video_id: str

    label: int

    frames: List[PoseResult] = field(default_factory=list)


# ==========================================================
# Frame Result
# ==========================================================

@dataclass
class FrameResult:
    """
    Intermediate representation of one processed frame.
    """

    frame_index: int

    detections: List[Detection] = field(default_factory=list)

    tracks: List[Track] = field(default_factory=list)

    poses: List[PoseResult] = field(default_factory=list)

@dataclass
class VideoResult:
    """
    Results for one processed video.
    """

    video_id: str
    label: int

    width: int
    height: int

    total_frames: int

    frames: List[FrameResult] = field(default_factory=list)
