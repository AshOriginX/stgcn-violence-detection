"""
Build college_violence PKLs using the predefined train/val split.

Does NOT modify existing RWF2000/RLVS/HOCKEYFIGHT PKLs.
"""

import argparse
import csv
import logging
import pickle
import sys
import time
from pathlib import Path

import cv2
import yaml

cv2.setNumThreads(1)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.detector import YOLODetector
from pipeline.exporter import PipelineExporter
from pipeline.extractor import PipelineExtractor
from pipeline.models import ModelFactory
from pipeline.pose import RTMPoseEstimator
from pipeline.tracker import ByteTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LABEL_MAP = {"Fight": 1, "NonFight": 0}


def create_manifest(college_dir: Path, output_csv: Path) -> list:
    """Scan college_violence/ and write manifest CSV."""
    entries = []
    for split in ("train", "val"):
        for label_name, label_id in LABEL_MAP.items():
            label_dir = college_dir / split / label_name
            if not label_dir.exists():
                continue
            for video_path in sorted(label_dir.glob("*.mp4")):
                entries.append({
                    "relative_path": str(video_path),
                    "video_id": video_path.stem,
                    "label_id": label_id,
                    "split": split,
                })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["relative_path", "video_id", "label_id", "split"]
        )
        writer.writeheader()
        writer.writerows(entries)

    logger.info("Created manifest with %d videos at %s", len(entries), output_csv)
    return entries


def build_split_pkls(annotations_by_split: dict, output_dir: Path):
    """Write train.pkl and val.pkl preserving predefined split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val"):
        anns = annotations_by_split.get(split_name, [])
        pyskl_dict = {
            "split": {split_name: [a["frame_dir"] for a in anns]},
            "annotations": anns,
        }
        out_path = output_dir / f"{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(pyskl_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote %s (%d annotations)", out_path, len(anns))


def main():
    parser = argparse.ArgumentParser(description="Build college_violence PKLs")
    parser.add_argument(
        "--college-dir", type=Path, default=Path("college_violence")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/pkl/college")
    )
    parser.add_argument(
        "--annotation-dir", type=Path, default=Path("outputs/annotations/college")
    )
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("configs/pipeline.yaml")
    )
    args = parser.parse_args()

    manifest_path = args.output_dir / "college_manifest.csv"
    entries = create_manifest(args.college_dir, manifest_path)

    if not entries:
        logger.error("No college videos found")
        sys.exit(1)

    with open(args.pipeline_config) as f:
        pipeline_config = yaml.safe_load(f)

    # Check already-processed
    args.annotation_dir.mkdir(parents=True, exist_ok=True)
    completed = {p.stem for p in args.annotation_dir.glob("*.pkl")}
    logger.info("Found %d already-processed college videos", len(completed))

    annotations_by_split = {"train": [], "val": []}
    for p in args.annotation_dir.glob("*.pkl"):
        with open(p, "rb") as f:
            ann = pickle.load(f)
        split = ann.get("split_tag", "train")
        annotations_by_split[split].append(ann)

    # Initialize pipeline
    model_factory = ModelFactory()
    detector_model = model_factory.load_yolo(pipeline_config["detector"])
    pose_model = model_factory.load_mmpose(pipeline_config["pose"])

    detector = YOLODetector(detector_model, pipeline_config["detector"])
    tracker = ByteTracker(detector_model, pipeline_config["tracker"])
    pose_estimator = RTMPoseEstimator(pose_model, pipeline_config["pose"])

    extractor = PipelineExtractor(
        detector, tracker, pose_estimator,
        {
            "skip_frames": pipeline_config["general"]["skip_frames"],
            "max_frames": pipeline_config["general"]["max_frames"],
        },
    )
    exporter = PipelineExporter(
        max_tracks=pipeline_config["general"].get("max_tracks", 2),
        num_keypoints=pipeline_config["general"].get("num_keypoints", 17),
    )

    to_process = [e for e in entries if e["video_id"] not in completed]
    logger.info("Processing %d new videos...", len(to_process))

    for i, entry in enumerate(to_process):
        logger.info(
            "Processing %d/%d: %s", i + 1, len(to_process), entry["video_id"]
        )
        try:
            video_result = extractor.process_video(
                Path(entry["relative_path"]),
                entry["video_id"],
                entry["label_id"],
            )
            annotation = exporter.export(video_result)
            annotation["split_tag"] = entry["split"]

            ann_path = args.annotation_dir / f"{entry['video_id']}.pkl"
            with open(ann_path, "wb") as f:
                pickle.dump(annotation, f)

            annotations_by_split[entry["split"]].append(annotation)
        except Exception as e:
            logger.error("Failed %s: %s", entry["video_id"], e)

    build_split_pkls(annotations_by_split, args.output_dir)

    extractor.close()
    exporter.close()
    logger.info("College dataset build complete!")


if __name__ == "__main__":
    main()
