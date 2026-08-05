"""
Build dataset script for preprocessing videos into PySKL training format.

This script orchestrates the entire preprocessing pipeline:
1. Load configuration from datasets.yaml
2. Read dataset manifest CSV
3. Initialize models (detector, tracker, pose_estimator)
4. Process each video through the pipeline
5. Export annotations and build train/val PKL files
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import yaml

from pipeline.dataset_builder import DatasetBuilder
from pipeline.detector import YOLODetector
from pipeline.exporter import PipelineExporter
from pipeline.extractor import PipelineExtractor
from pipeline.models import ModelFactory
from pipeline.pose import RTMPoseEstimator
from pipeline.tracker import ByteTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def read_manifest(manifest_path: Path) -> list:
    """
    Read dataset manifest CSV file.

    Expected CSV format:
    video_path,video_id,label

    Parameters
    ----------
    manifest_path : Path
        Path to the manifest CSV file.

    Returns
    -------
    list
        List of dictionaries with keys: video_path, video_id, label.
    """

    entries = []

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                {
                    "video_path": Path(row["video_path"]),
                    "video_id": row["video_id"],
                    "label": int(row["label"]),
                }
            )

    logger.info("Read %d entries from manifest %s", len(entries), manifest_path)
    return entries


def main():
    """Main entry point for dataset building."""

    parser = argparse.ArgumentParser(description="Build PySKL training dataset from videos")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Path to datasets configuration file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to dataset manifest CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pkl"),
        help="Directory to save PKL files",
    )

    args = parser.parse_args()

    # Load configurations
    pipeline_config = load_config(args.config)
    datasets_config = load_config(args.datasets_config)

    # Read manifest
    manifest_entries = read_manifest(args.manifest)

    if not manifest_entries:
        logger.error("No entries found in manifest")
        sys.exit(1)

    # Initialize models
    logger.info("Initializing models...")

    model_factory = ModelFactory()

    detector_model = model_factory.load_yolo(pipeline_config["detector"])

    # Reuse detector model for tracker (ByteTrack uses same YOLO model)
    tracker_model = detector_model

    pose_model = model_factory.load_mmpose(pipeline_config["pose"])

    # Initialize pipeline components
    logger.info("Initializing pipeline components...")

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

    # Initialize dataset builder
    dataset_builder = DatasetBuilder(
        output_dir=args.output_dir,
        val_split=pipeline_config["general"].get("val_split", 0.2),
        seed=datasets_config["seed"],
    )

    # Create failed videos log file
    failed_videos_path = args.output_dir.parent / "reports" / "failed_videos.txt"
    failed_videos_path.parent.mkdir(parents=True, exist_ok=True)

    # Process videos
    logger.info("Processing %d videos...", len(manifest_entries))

    for i, entry in enumerate(manifest_entries):
        logger.info(
            "Processing video %d/%d: %s",
            i + 1,
            len(manifest_entries),
            entry["video_id"],
        )

        try:
            # Process video through pipeline
            video_result = extractor.process_video(
                entry["video_path"],
                entry["video_id"],
                entry["label"],
            )

            # Export to PySKL annotation format
            annotation = exporter.export(video_result)

            # Add to dataset builder
            dataset_builder.add(annotation)

        except Exception as e:
            logger.error("Failed to process video %s: %s", entry["video_id"], e)
            with open(failed_videos_path, "a") as f:
                f.write(f"{entry['video_id']}: {e}\n")
            continue

    # Build dataset
    logger.info("Building train/val PKL files...")
    if dataset_builder.annotations:
        dataset_builder.build()
    else:
        logger.warning("No annotations to build. Skipping dataset build.")

    # Cleanup
    logger.info("Cleaning up resources...")
    extractor.close()
    exporter.close()
    dataset_builder.close()

    logger.info("Dataset building complete!")


if __name__ == "__main__":
    main()
