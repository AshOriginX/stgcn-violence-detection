"""
Video utilities for the ST-GCN++ preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict

import cv2


def read_video_metadata(video_path: Path) -> Dict:
    """
    Read metadata from a video.

    Parameters
    ----------
    video_path : Path

    Returns
    -------
    dict
        Video metadata and validation status.
    """

    video_path = Path(video_path)

    metadata = {
        "path": str(video_path),
        "filename": video_path.name,
        "readable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration": 0.0,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 3),
        "error": ""
    }

    if not video_path.exists():
        metadata["error"] = "File not found"
        return metadata

    if video_path.stat().st_size == 0:
        metadata["error"] = "Zero-byte file"
        return metadata

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        metadata["error"] = "Cannot open video"
        return metadata

    success, _ = cap.read()

    if not success:
        metadata["error"] = "Cannot decode first frame"
        cap.release()
        return metadata

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = 0.0
    if fps > 0:
        duration = frame_count / fps

    metadata.update({
        "readable": True,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2)
    })

    cap.release()

    return metadata

def open_video(video_path: Path) -> cv2.VideoCapture:
    """
    Open a video file and return an initialized VideoCapture object.

    Parameters
    ----------
    video_path : Path

    Returns
    -------
    cv2.VideoCapture
        OpenCV video capture object.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.

    RuntimeError
        If OpenCV cannot open the video.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video:\n{video_path}")

    return cap
