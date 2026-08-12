"""
Visualize skeleton keypoints on original video frames.

This script loads a PySKL annotation PKL file and overlays the detected
skeleton keypoints and connections on the original video frames for validation.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

# COCO 17 keypoints skeleton connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # upper body
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # lower body
]

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Colors for visualization
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (0, 255, 255)  # Yellow
TEXT_COLOR = (255, 255, 255)  # White


def draw_skeleton(frame, keypoints, keypoint_scores, threshold=0.5):
    """
    Draw skeleton keypoints and connections on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame in BGR format.
    keypoints : np.ndarray
        Keypoint array of shape (M, K, 2) where M is max tracks,
        K is num keypoints (17).
    keypoint_scores : np.ndarray
        Keypoint scores of shape (M, K).
    threshold : float
        Minimum score threshold to draw a keypoint.

    Returns
    -------
    np.ndarray
        Frame with drawn skeleton.
    """
    frame = frame.copy()

    num_tracks, num_keypoints, _ = keypoints.shape

    # Draw skeleton connections
    for track_idx in range(num_tracks):
        for (kp1_idx, kp2_idx) in SKELETON_CONNECTIONS:
            pt1 = keypoints[track_idx, kp1_idx]
            pt2 = keypoints[track_idx, kp2_idx]
            score1 = keypoint_scores[track_idx, kp1_idx]
            score2 = keypoint_scores[track_idx, kp2_idx]

            # Check if both keypoints are valid
            if score1 > threshold and score2 > threshold:
                pt1 = tuple(pt1.astype(int))
                pt2 = tuple(pt2.astype(int))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)

    # Draw keypoints
    for track_idx in range(num_tracks):
        for kp_idx in range(num_keypoints):
            kp = keypoints[track_idx, kp_idx]
            score = keypoint_scores[track_idx, kp_idx]

            if score > threshold:
                kp = tuple(kp.astype(int))
                cv2.circle(frame, kp, 4, KEYPOINT_COLOR, -1)

    return frame


def visualize_skeleton(annotation_path, video_path, output_path=None):
    """
    Visualize skeleton keypoints on original video.

    Parameters
    ----------
    annotation_path : str or Path
        Path to annotation PKL file.
    video_path : str or Path
        Path to original video file.
    output_path : str or Path, optional
        Path to save output video. If None, displays in window.
    """
    # Load annotation
    with open(annotation_path, "rb") as f:
        annotation = pickle.load(f)

    print(f"Loaded annotation: {annotation['frame_dir']}")
    print(f"Keypoint shape: {annotation['keypoint'].shape}")
    print(f"Keypoint score shape: {annotation['keypoint_score'].shape}")
    print(f"Total frames: {annotation['total_frames']}")
    print(f"Label: {annotation['label']}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    else:
        out = None

    keypoints = annotation["keypoint"]
    keypoint_scores = annotation["keypoint_score"]
    total_frames = annotation["total_frames"]

    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw skeleton for current frame
        # Use the corresponding frame index from keypoints
        if frame_idx < keypoints.shape[1]:
            frame_keypoints = keypoints[:, frame_idx]
            frame_scores = keypoint_scores[:, frame_idx]
            frame = draw_skeleton(frame, frame_keypoints, frame_scores)

        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Label: {annotation['label']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

        # Write or display
        if out:
            out.write(frame)
        else:
            cv2.imshow('Skeleton Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames")


def main():
    parser = argparse.ArgumentParser(description="Visualize skeleton keypoints on video")
    parser.add_argument("--annotation", required=True, help="Path to annotation PKL file")
    parser.add_argument("--video", required=True, help="Path to original video file")
    parser.add_argument("--output", help="Path to save output video (optional)")
    args = parser.parse_args()

    visualize_skeleton(args.annotation, args.video, args.output)


if __name__ == "__main__":
    main()
