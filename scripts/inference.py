"""
Inference script for ST-GCN++ violence detection on raw videos.

This script performs end-to-end inference:
video → pose extraction → preprocessing → ST-GCN++ prediction → aggregation
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Limit OpenCV thread usage
cv2.setNumThreads(1)

# COCO 17 keypoints skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # upper body
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # lower body
]

# Colors for visualization
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (0, 255, 255)  # Yellow
TEXT_COLOR = (255, 255, 255)  # White
FIGHT_COLOR = (0, 0, 255)  # Red
NONFIGHT_COLOR = (0, 255, 0)  # Green
MONITORING_COLOR = (0, 200, 255)  # Yellow/amber
MODERATE_COLOR = (0, 165, 255)  # Orange


@dataclass
class TemporalDecisionConfig:
    """Configurable thresholds for temporal fight detection."""

    monitoring_low_threshold: float = 0.55
    fight_threshold: float = 0.65
    min_consecutive_fight_windows: int = 2
    attribution_threshold: float = 0.3
    attribution_contribution_threshold: float = 0.15
    min_consecutive_attribution_windows: int = 2


def classify_window_state(fight_prob: float, config: TemporalDecisionConfig) -> str:
    """Classify a single window as non_fight, monitoring, or candidate_fight."""
    if fight_prob < config.monitoring_low_threshold:
        return "non_fight"
    if fight_prob < config.fight_threshold:
        return "monitoring"
    return "candidate_fight"


def compute_confirmed_fight_windows(
    fight_probs: List[float],
    config: TemporalDecisionConfig,
) -> Set[int]:
    """
    Identify confirmed fight windows requiring consecutive elevated probability.

    Falls back to the strongest single window when fewer windows exist than
    min_consecutive_fight_windows.
    """
    n = len(fight_probs)
    if n == 0:
        return set()

    confirmed: Set[int] = set()
    min_run = config.min_consecutive_fight_windows

    # Find runs of candidate-or-higher windows (prob >= fight_threshold)
    run_start = None
    for i, prob in enumerate(fight_probs):
        is_candidate = prob >= config.fight_threshold
        if is_candidate:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_run:
                    confirmed.update(range(run_start, i))
                run_start = None
    if run_start is not None:
        run_len = n - run_start
        if run_len >= min_run:
            confirmed.update(range(run_start, n))

    # Graceful fallback for short videos
    if not confirmed and n < min_run:
        best_idx = max(range(n), key=lambda i: fight_probs[i])
        if fight_probs[best_idx] >= config.fight_threshold:
            confirmed.add(best_idx)

    return confirmed


def compute_stable_participants(
    confirmed_indices: List[int],
    window_attributions: List[List[float]],
    config: TemporalDecisionConfig,
) -> Tuple[Set[int], bool]:
    """
    Find participants with stable attribution across consecutive fight windows.

    Returns
    -------
    Tuple[Set[int], bool]
        Stable participant indices and whether attribution is uncertain.
    """
    if not confirmed_indices:
        return set(), True

    high_per_window: List[Set[int]] = []
    for idx in confirmed_indices:
        attrs = window_attributions[idx] if idx < len(window_attributions) else []
        high = {
            i for i, a in enumerate(attrs)
            if a > config.attribution_contribution_threshold
        }
        high_per_window.append(high)

    if len(confirmed_indices) < config.min_consecutive_attribution_windows:
        return set(), True

    max_persons = max(
        (len(window_attributions[i]) for i in confirmed_indices if i < len(window_attributions)),
        default=2,
    )
    stable: Set[int] = set()
    for person_idx in range(max_persons):
        consecutive = 0
        max_consecutive = 0
        for high in high_per_window:
            if person_idx in high:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        if max_consecutive >= config.min_consecutive_attribution_windows:
            stable.add(person_idx)

    uncertain = len(stable) == 0
    return stable, uncertain


def build_temporal_decisions(
    fight_probs: List[float],
    window_attributions: List[List[float]],
    config: TemporalDecisionConfig,
) -> Dict:
    """Build complete temporal decision results for all windows."""
    confirmed = compute_confirmed_fight_windows(fight_probs, config)
    confirmed_sorted = sorted(confirmed)

    stable_participants, attribution_uncertain = compute_stable_participants(
        confirmed_sorted, window_attributions, config
    )

    window_states = []
    for i, prob in enumerate(fight_probs):
        state = classify_window_state(prob, config)
        is_confirmed = i in confirmed
        window_states.append({
            "window_idx": i,
            "fight_probability": prob,
            "state": state,
            "confirmed_fight": is_confirmed,
        })

    if confirmed:
        final_state = "confirmed_fight"
    elif any(s["state"] == "monitoring" for s in window_states):
        final_state = "monitoring"
    else:
        final_state = "non_fight"

    return {
        "window_states": window_states,
        "confirmed_fight_windows": sorted(confirmed),
        "stable_participants": sorted(stable_participants),
        "attribution_uncertain": attribution_uncertain,
        "final_state": final_state,
    }

from pipeline.detector import YOLODetector
from pipeline.exporter import PipelineExporter
from pipeline.extractor import PipelineExtractor
from pipeline.models import ModelFactory
from pipeline.pose import RTMPoseEstimator
from pipeline.tracker import ByteTracker
from training.model import STGCNPlusPlus
from training.preprocessing import (
    WindowIndex,
    generate_windows,
    preprocess_window,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def draw_skeleton(frame, keypoints, keypoint_scores, threshold=0.5, person_colors=None):
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
    person_colors : List[Tuple[int, int, int]], optional
        Colors for each person (M colors). If None, uses default green.

    Returns
    -------
    np.ndarray
        Frame with drawn skeleton.
    """
    frame = frame.copy()

    num_tracks, num_keypoints, _ = keypoints.shape

    # Use default green color if no person colors provided
    if person_colors is None:
        person_colors = [KEYPOINT_COLOR] * num_tracks

    # Draw skeleton connections
    for track_idx in range(num_tracks):
        track_color = person_colors[track_idx] if track_idx < len(person_colors) else KEYPOINT_COLOR
        for (kp1_idx, kp2_idx) in SKELETON_CONNECTIONS:
            pt1 = keypoints[track_idx, kp1_idx]
            pt2 = keypoints[track_idx, kp2_idx]
            score1 = keypoint_scores[track_idx, kp1_idx]
            score2 = keypoint_scores[track_idx, kp2_idx]

            # Check if both keypoints are valid
            if score1 > threshold and score2 > threshold:
                pt1 = tuple(pt1.astype(int))
                pt2 = tuple(pt2.astype(int))
                cv2.line(frame, pt1, pt2, track_color, 2)

    # Draw keypoints
    for track_idx in range(num_tracks):
        track_color = person_colors[track_idx] if track_idx < len(person_colors) else KEYPOINT_COLOR
        for kp_idx in range(num_keypoints):
            kp = keypoints[track_idx, kp_idx]
            score = keypoint_scores[track_idx, kp_idx]

            if score > threshold:
                kp = tuple(kp.astype(int))
                cv2.circle(frame, kp, 4, track_color, -1)

    return frame


def compute_person_attribution(
    model: torch.nn.Module,
    window_tensor: torch.Tensor,
    full_fight_prob: float,
    device: torch.device,
    attribution_threshold: float = 0.3,
) -> List[float]:
    """
    Compute leave-one-person-out attribution for each person in a window.

    Parameters
    ----------
    model : torch.nn.Module
        ST-GCN++ model.
    window_tensor : torch.Tensor
        Window tensor of shape (C, T, V, M) where M is number of persons.
    full_fight_prob : float
        Fight probability with all persons.
    device : torch.device
        Computation device.
    attribution_threshold : float
        Minimum fight probability to perform attribution.

    Returns
    -------
    List[float]
        Attribution scores for each person (contribution to fight probability).
    """
    # Only perform attribution if fight probability exceeds threshold
    if full_fight_prob < attribution_threshold:
        return []

    num_persons = window_tensor.shape[3]  # M dimension
    attributions = []

    # Compute contribution for each person
    for person_idx in range(num_persons):
        # Create copy with this person zeroed
        window_copy = window_tensor.clone()
        window_copy[:, :, :, person_idx] = 0

        # Run inference without this person
        window_copy = window_copy.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            logits = model(window_copy)
            prob_without_person = torch.softmax(logits, dim=1)[0, 1].item()

        # Contribution = full_prob - prob_without_person
        contribution = full_fight_prob - prob_without_person
        attributions.append(contribution)

    return attributions


def render_annotated_video(
    video_path: Path,
    video_result,
    windows: List[WindowIndex],
    window_predictions: List[float],
    output_path: Path,
    temporal_decisions: Dict,
    window_attributions: List[List[float]] = None,
    config: TemporalDecisionConfig = None,
) -> Path:
    """
    Render annotated video with skeleton overlay and ST-GCN++ predictions.

    Parameters
    ----------
    video_path : Path
        Path to original video file.
    video_result : VideoResult
        VideoResult from pose extraction.
    windows : List[WindowIndex]
        Window indices from inference.
    window_predictions : List[float]
        Fight probabilities for each window.
    output_path : Path
        Path to save output video.
    window_attributions : List[List[float]], optional
        Attribution scores for each person in each window.
    temporal_decisions : Dict
        Output from build_temporal_decisions().
    config : TemporalDecisionConfig, optional
        Decision configuration for attribution thresholds.

    Returns
    -------
    Path
        Path to saved video.
    """
    logger.info(f"Rendering annotated video to {output_path}")

    if config is None:
        config = TemporalDecisionConfig()

    confirmed_windows = set(temporal_decisions["confirmed_fight_windows"])
    stable_participants = set(temporal_decisions["stable_participants"])
    attribution_uncertain = temporal_decisions["attribution_uncertain"]
    window_state_map = {
        ws["window_idx"]: ws for ws in temporal_decisions["window_states"]
    }

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Build frame to window mapping
    frame_to_window = {}
    for window_idx, window in enumerate(windows):
        for frame_idx in range(window.window_start, window.window_end):
            if frame_idx not in frame_to_window:
                frame_to_window[frame_idx] = window_idx

    # Get keypoint data from video_result
    # video_result has frame_results list with FrameResult objects
    # Each FrameResult has poses (list of PoseResult)
    # We need to extract keypoints and scores for each frame

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw skeleton if pose data available for this frame
        person_colors = None
        if frame_idx < len(video_result.frames):
            frame_result = video_result.frames[frame_idx]

            # Build keypoint arrays for this frame
            # Max 2 tracks, 17 keypoints
            keypoints = np.zeros((2, 17, 2), dtype=np.float32)
            keypoint_scores = np.zeros((2, 17), dtype=np.float32)

            for pose_idx, pose in enumerate(frame_result.poses[:2]):  # Max 2 tracks
                if pose.keypoints is not None and len(pose.keypoints) == 17:
                    keypoints[pose_idx] = pose.keypoints
                    keypoint_scores[pose_idx] = pose.scores if pose.scores is not None else np.ones(17)

        # Get ST-GCN++ prediction for current frame
        if frame_idx in frame_to_window:
            window_idx = frame_to_window[frame_idx]
            window = windows[window_idx]
            fight_prob = window_predictions[window_idx]
            ws = window_state_map.get(window_idx, {})
            is_confirmed = window_idx in confirmed_windows
            state = ws.get("state", "non_fight")

            # Determine display state and colors
            if is_confirmed:
                pred_color = FIGHT_COLOR
                pred_text = "FIGHT"
            elif state == "monitoring":
                pred_color = MONITORING_COLOR
                pred_text = "MONITORING"
            else:
                pred_color = NONFIGHT_COLOR
                pred_text = "NON-FIGHT"

            # Attribution only for confirmed fight windows
            attributions = None
            if is_confirmed and window_attributions and window_idx < len(window_attributions):
                attributions = window_attributions[window_idx]

            # Person colors: red only for stable participants in confirmed fight
            person_colors = None
            if is_confirmed and attributions:
                person_colors = []
                for person_idx, attr in enumerate(attributions):
                    if (
                        not attribution_uncertain
                        and person_idx in stable_participants
                    ):
                        person_colors.append(FIGHT_COLOR)
                    elif attr > config.attribution_contribution_threshold:
                        person_colors.append(MODERATE_COLOR)
                    else:
                        person_colors.append(KEYPOINT_COLOR)
            elif not is_confirmed:
                person_colors = None

            # Draw skeleton with attribution colors
            if frame_idx < len(video_result.frames):
                frame = draw_skeleton(
                    frame, keypoints, keypoint_scores, person_colors=person_colors
                )

            # Draw status panel
            panel_height = 120 if is_confirmed else 90
            cv2.rectangle(
                frame, (10, height - panel_height), (420, height - 10), pred_color, -1
            )

            cv2.putText(
                frame, f"ST-GCN++: {pred_text}", (20, height - panel_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2,
            )
            cv2.putText(
                frame, f"Fight Prob: {fight_prob:.3f}", (20, height - panel_height + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2,
            )

            # Participant line only for confirmed fight
            if is_confirmed:
                if attribution_uncertain or not stable_participants:
                    participant_text = "Participants: uncertain"
                else:
                    p_labels = [f"P{i + 1}" for i in sorted(stable_participants)]
                    participant_text = f"Suspected participants: {', '.join(p_labels)}"
                cv2.putText(
                    frame, participant_text, (20, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2,
                )
        else:
            # Frame outside any window (padding frames)
            if frame_idx < len(video_result.frames):
                frame = draw_skeleton(frame, keypoints, keypoint_scores)
            cv2.rectangle(frame, (10, height - 60), (250, height - 10), (100, 100, 100), -1)
            cv2.putText(frame, "No Window", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

        # Write frame
        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    logger.info(f"Rendered {frame_idx} frames to {output_path}")
    return output_path


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_checkpoint(checkpoint_path: Path) -> Tuple[dict, torch.nn.Module]:
    """
    Load checkpoint and model.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file.

    Returns
    -------
    Tuple[dict, torch.nn.Module]
        Checkpoint metadata and loaded model.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    logger.info(f"Checkpoint metadata:")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Best F1: {checkpoint['best_metric']:.4f}")

    # Load config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        logger.info(f"Checkpoint config:")
        logger.info(f"  tcn_dropout: {config.get('tcn_dropout', 'N/A')}")
        logger.info(f"  scheduler: {config.get('scheduler', 'N/A')}")

    # Initialize model with Phase-4 parameters
    model = STGCNPlusPlus(num_classes=2, tcn_dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint, model


def initialize_pipeline(pipeline_config: dict):
    """
    Initialize pose extraction pipeline components.

    Parameters
    ----------
    pipeline_config : dict
        Pipeline configuration dictionary.

    Returns
    -------
    Tuple[PipelineExtractor, PipelineExporter]
        Initialized extractor and exporter.
    """
    logger.info("Initializing pose extraction pipeline...")

    model_factory = ModelFactory()

    # Load models
    detector_model = model_factory.load_yolo(pipeline_config["detector"])
    tracker_model = detector_model  # Reuse YOLO for ByteTrack
    pose_model = model_factory.load_mmpose(pipeline_config["pose"])

    # Initialize components
    detector = YOLODetector(detector_model, pipeline_config["detector"])
    tracker = ByteTracker(tracker_model, pipeline_config["tracker"])
    pose_estimator = RTMPoseEstimator(pose_model, pipeline_config["pose"])

    # Initialize extractor
    extractor_config = {
        "skip_frames": pipeline_config["general"]["skip_frames"],
        "max_frames": pipeline_config["general"]["max_frames"],
    }
    extractor = PipelineExtractor(detector, tracker, pose_estimator, extractor_config)

    # Initialize exporter
    exporter = PipelineExporter(
        max_tracks=pipeline_config["general"].get("max_tracks", 2),
        num_keypoints=pipeline_config["general"].get("num_keypoints", 17),
    )

    logger.info("Pose extraction pipeline initialized")
    return extractor, exporter


def preprocess_for_inference(
    keypoint: np.ndarray,
    clip_len: int = 150,
    window_stride: int = 75,
    enable_normalization: bool = True,
    max_windows_per_video: int = None,
) -> Tuple[List[WindowIndex], np.ndarray]:
    """
    Preprocess keypoint array for inference.

    Parameters
    ----------
    keypoint : np.ndarray
        Keypoint array with shape (M, T, V, C).
    clip_len : int
        Target window length.
    window_stride : int
        Window stride for extraction.
    enable_normalization : bool
        Whether to apply spatial normalization.
    max_windows_per_video : int, optional
        Maximum windows per video.

    Returns
    -------
    Tuple[List[WindowIndex], np.ndarray]
        Window indices and preprocessed window tensors.
    """
    # Generate windows
    windows = generate_windows(
        keypoint=keypoint,
        video_id="inference",
        label=0,  # Dummy label for inference
        clip_len=clip_len,
        window_stride=window_stride,
        enable_windowing=True,
        max_windows_per_video=max_windows_per_video,
    )

    # Preprocess each window
    preprocessed_windows = []
    for window in windows:
        window_keypoint = keypoint[:, window.window_start:window.window_end, :, :]
        preprocessed = preprocess_window(
            keypoint=window_keypoint,
            clip_len=clip_len,
            enable_normalization=enable_normalization,
            invalid_value_policy="zero",
        )
        preprocessed_windows.append(preprocessed)

    return windows, np.array(preprocessed_windows)


def run_inference_on_video(
    video_path: Path,
    model: torch.nn.Module,
    extractor: PipelineExtractor,
    exporter: PipelineExporter,
    clip_len: int = 150,
    window_stride: int = 75,
    enable_normalization: bool = True,
    device: torch.device = torch.device("cpu"),
    save_video: bool = False,
    output_dir: Path = None,
    decision_config: TemporalDecisionConfig = None,
) -> Dict:
    """
    Run inference on a single video.

    Parameters
    ----------
    video_path : Path
        Path to video file.
    model : torch.nn.Module
        Loaded ST-GCN++ model.
    extractor : PipelineExtractor
        Pose extraction pipeline.
    exporter : PipelineExporter
        Pipeline exporter.
    clip_len : int
        Target window length.
    window_stride : int
        Window stride.
    enable_normalization : bool
        Whether to apply normalization.
    device : torch.device
        Computation device.
    save_video : bool
        Whether to save annotated output video.
    output_dir : Path
        Directory to save output video.

    Returns
    -------
    Dict
        Inference results dictionary.
    """
    if decision_config is None:
        decision_config = TemporalDecisionConfig()

    video_id = video_path.stem
    logger.info(f"Processing video: {video_path}")

    start_time = time.time()

    # Extract poses
    try:
        video_result = extractor.process_video(video_path, video_id, label=0)
    except Exception as e:
        logger.error(f"Pose extraction failed for {video_id}: {e}")
        return {
            "video_id": video_id,
            "success": False,
            "error": str(e),
        }

    pose_time = time.time() - start_time
    logger.info(f"Pose extraction completed in {pose_time:.2f}s")

    # Export to keypoint tensor
    try:
        annotation = exporter.export(video_result)
    except Exception as e:
        logger.error(f"Export failed for {video_id}: {e}")
        return {
            "video_id": video_id,
            "success": False,
            "error": str(e),
        }

    keypoint = annotation["keypoint"]
    total_frames = annotation["total_frames"]
    logger.info(f"Keypoint shape: {keypoint.shape}, Total frames: {total_frames}")

    # Preprocess for inference
    windows, preprocessed_windows = preprocess_for_inference(
        keypoint=keypoint,
        clip_len=clip_len,
        window_stride=window_stride,
        enable_normalization=enable_normalization,
    )

    logger.info(f"Generated {len(windows)} windows")

    # Convert to tensor format (N, C, T, V, M)
    # preprocessed_windows: (N, M, T, V, C) -> (N, C, T, V, M)
    preprocessed_windows = torch.from_numpy(preprocessed_windows).float()
    preprocessed_windows = preprocessed_windows.permute(0, 4, 2, 3, 1)  # (N, C, T, V, M)

    # Log checksums for each window to detect duplicates
    logger.info(f"Window tensor checksums:")
    for i, (window, tensor) in enumerate(zip(windows, preprocessed_windows)):
        checksum = hash(tensor.data_ptr())  # Hash of memory address
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        logger.info(f"  Window {i} [{window.window_start}:{window.window_end}]: "
                   f"checksum={checksum}, mean={mean_val:.6f}, std={std_val:.6f}")

    # Check for identical tensors
    for i in range(len(preprocessed_windows)):
        for j in range(i + 1, len(preprocessed_windows)):
            if torch.equal(preprocessed_windows[i], preprocessed_windows[j]):
                logger.warning(f"WARNING: Window {i} and Window {j} have IDENTICAL tensors!")

    # Run inference
    model.to(device)
    preprocessed_windows = preprocessed_windows.to(device)

    with torch.no_grad():
        logits = model(preprocessed_windows)  # (N, 2)
        probabilities = torch.softmax(logits, dim=1)  # (N, 2)
        predictions = logits.argmax(dim=1)  # (N,)

    # Move to CPU for processing
    probabilities = probabilities.cpu().numpy()
    predictions = predictions.cpu().numpy()
    preprocessed_windows_cpu = preprocessed_windows.cpu()

    # Temporal decision logic first (before expensive attribution)
    fight_probs_list = probabilities[:, 1].tolist()
    temporal_decisions = build_temporal_decisions(
        fight_probs_list, [], decision_config
    )

    # Compute person attribution ONLY for confirmed fight windows
    window_attributions: List[List[float]] = []
    confirmed_windows = set(temporal_decisions["confirmed_fight_windows"])

    if confirmed_windows and temporal_decisions["final_state"] == "confirmed_fight":
        for i, (window_tensor, fight_prob) in enumerate(
            zip(preprocessed_windows_cpu, fight_probs_list)
        ):
            if i in confirmed_windows and fight_prob >= decision_config.attribution_threshold:
                attributions = compute_person_attribution(
                    model=model,
                    window_tensor=window_tensor,
                    full_fight_prob=fight_prob,
                    device=device,
                    attribution_threshold=decision_config.attribution_threshold,
                )
                window_attributions.append(attributions)
            else:
                window_attributions.append([])
    else:
        # No attribution needed for non-confirmed fights
        window_attributions = [[] for _ in fight_probs_list]

    # Aggregate results
    fight_probs = probabilities[:, 1]
    avg_fight_prob = fight_probs.mean()
    max_fight_prob = fight_probs.max()
    final_state = temporal_decisions["final_state"]
    final_prediction = 1 if final_state == "confirmed_fight" else 0

    # Build window-level results
    window_results = []
    for i, (window, pred, prob) in enumerate(zip(windows, predictions, fight_probs)):
        ws = temporal_decisions["window_states"][i]
        window_results.append({
            "window_idx": i,
            "start_frame": window.window_start,
            "end_frame": window.window_end,
            "prediction": int(pred),
            "fight_probability": float(prob),
            "state": ws["state"],
            "confirmed_fight": ws["confirmed_fight"],
        })

    total_time = time.time() - start_time

    # Render annotated video if requested
    video_output_path = None
    if save_video:
        # Create results directory at outputs/results/
        results_dir = Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)

        video_output_path = results_dir / f"{video_id}_annotated.mp4"

        try:
            render_annotated_video(
                video_path=video_path,
                video_result=video_result,
                windows=windows,
                window_predictions=fight_probs.tolist(),
                output_path=video_output_path,
                temporal_decisions=temporal_decisions,
                window_attributions=window_attributions,
                config=decision_config,
            )
        except Exception as e:
            logger.error(f"Video rendering failed for {video_id}: {e}")
            video_output_path = None

    return {
        "video_id": video_id,
        "success": True,
        "total_frames": total_frames,
        "num_windows": len(windows),
        "window_results": window_results,
        "avg_fight_probability": float(avg_fight_prob),
        "max_fight_probability": float(max_fight_prob),
        "final_prediction": int(final_prediction),
        "final_state": final_state,
        "confirmed_fight_windows": temporal_decisions["confirmed_fight_windows"],
        "stable_participants": temporal_decisions["stable_participants"],
        "attribution_uncertain": temporal_decisions["attribution_uncertain"],
        "confidence": float(max(avg_fight_prob, 1 - avg_fight_prob)),
        "pose_extraction_time": pose_time,
        "total_time": total_time,
        "video_output_path": str(video_output_path) if video_output_path else None,
    }


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(
        description="ST-GCN++ inference on raw videos"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pt file)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=False,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=False,
        help="Single video file to process (alternative to --input-dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Path to pipeline configuration"
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=150,
        help="Temporal window length"
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=75,
        help="Window stride for extraction"
    )
    parser.add_argument(
        "--enable-normalization",
        action="store_true",
        default=True,
        help="Enable spatial normalization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (cpu or cuda)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output video with skeleton overlay and predictions"
    )
    parser.add_argument(
        "--monitoring-low-threshold",
        type=float,
        default=0.55,
        help="Below this probability → NON-FIGHT (default: 0.55)",
    )
    parser.add_argument(
        "--fight-threshold",
        type=float,
        default=0.65,
        help="Candidate fight threshold; requires consecutive windows (default: 0.65)",
    )
    parser.add_argument(
        "--min-consecutive-fight-windows",
        type=int,
        default=2,
        help="Consecutive windows above fight-threshold for confirmed FIGHT (default: 2)",
    )
    parser.add_argument(
        "--attribution-threshold",
        type=float,
        default=0.3,
        help="Minimum fight prob to compute attribution (default: 0.3)",
    )
    parser.add_argument(
        "--attribution-contribution-threshold",
        type=float,
        default=0.15,
        help="Minimum attribution contribution to flag a participant (default: 0.15)",
    )
    parser.add_argument(
        "--min-consecutive-attribution-windows",
        type=int,
        default=2,
        help="Consecutive fight windows for stable participant highlight (default: 2)",
    )

    args = parser.parse_args()

    decision_config = TemporalDecisionConfig(
        monitoring_low_threshold=args.monitoring_low_threshold,
        fight_threshold=args.fight_threshold,
        min_consecutive_fight_windows=args.min_consecutive_fight_windows,
        attribution_threshold=args.attribution_threshold,
        attribution_contribution_threshold=args.attribution_contribution_threshold,
        min_consecutive_attribution_windows=args.min_consecutive_attribution_windows,
    )

    # Validate that either --input-dir or --video is provided
    if args.input_dir is None and args.video is None:
        parser.error("Either --input-dir or --video must be provided")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and model
    checkpoint, model = load_checkpoint(args.checkpoint)

    # Load pipeline config
    pipeline_config = load_config(args.pipeline_config)

    # Initialize pose extraction pipeline
    extractor, exporter = initialize_pipeline(pipeline_config)

    # Resolve device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    # Determine video files to process
    video_files = []
    if args.video is not None:
        if not args.video.exists():
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)
        video_files = [args.video]
    else:
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        for ext in video_extensions:
            video_files.extend(args.input_dir.glob(f"*{ext}"))

    if not video_files:
        logger.error("No video files found")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} video file(s) to process")

    # Run inference on each video
    all_results = []
    for video_path in sorted(video_files):
        result = run_inference_on_video(
            video_path=video_path,
            model=model,
            extractor=extractor,
            exporter=exporter,
            clip_len=args.clip_len,
            window_stride=args.window_stride,
            enable_normalization=args.enable_normalization,
            device=device,
            save_video=args.save_video,
            output_dir=args.output_dir,
            decision_config=decision_config,
        )
        all_results.append(result)

        # Print per-video summary
        if result["success"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"VIDEO: {result['video_id']}")
            logger.info(f"{'='*60}")
            logger.info(f"Total frames: {result['total_frames']}")
            logger.info(f"Number of windows: {result['num_windows']}")
            logger.info(f"Average Fight probability: {result['avg_fight_probability']:.4f}")
            logger.info(f"Max Fight probability: {result['max_fight_probability']:.4f}")
            logger.info(f"Final state: {result['final_state']}")
            logger.info(f"Confirmed fight windows: {result['confirmed_fight_windows']}")
            logger.info(f"Final prediction: {'Fight' if result['final_prediction'] == 1 else 'NonFight'}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info(f"Pose extraction time: {result['pose_extraction_time']:.2f}s")
            logger.info(f"Total time: {result['total_time']:.2f}s")
        else:
            logger.error(f"VIDEO {result['video_id']} FAILED: {result['error']}")

    # Save results
    results_path = args.output_dir / "inference_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Cleanup
    extractor.close()
    exporter.close()

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
