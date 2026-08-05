"""
Dataset manifest generation for the ST-GCN++ preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict, List

from pipeline.utils import (
    ensure_dir,
    load_yaml,
    list_video_files,
    save_csv,
    save_json,
)


def _infer_label(path: Path) -> str:
    """
    Infer class label from directory names.
    """

    parts = {p.lower() for p in path.parts}

    if "fight" in parts or "violence" in parts:
        return "Fight"

    return "NonFight"


def create_manifest(
    config_path: str = "configs/datasets.yaml",
) -> Dict:
    """
    Generate the master dataset manifest.

    Returns
    -------
    dict
        Summary statistics.
    """

    config = load_yaml(config_path)

    manifest_dir = Path(config["outputs"]["manifests"])
    ensure_dir(manifest_dir)

    rows: List[Dict] = []

    dataset_counts = {}
    total_videos = 0

    for dataset_key, dataset_info in config["datasets"].items():

        if dataset_key == "unified":
            continue

        dataset_name = dataset_info["name"]
        dataset_path = Path(dataset_info["path"])

        videos = list_video_files(
            dataset_path,
            config["video_extensions"]
        )

        dataset_counts[dataset_name] = len(videos)

        for index, video in enumerate(videos, start=1):

            label = _infer_label(video)

            rows.append({

                "video_id":
                    f"{dataset_key.upper()}_{index:06d}",

                "source_index":
                    index,

                "dataset":
                    dataset_name,

                "label":
                    label,

                "label_id":
                    config["labels"][label],

                "filename":
                    video.name,

                "relative_path":
                    str(video),

                "extension":
                    video.suffix,

                "split":
                    "",

                "skeleton_path":
                    "",

                "num_persons":
                    0,

                "processing_status":
                    "pending",

            })

            total_videos += 1

    csv_path = manifest_dir / "dataset_manifest.csv"
    json_path = manifest_dir / "dataset_manifest.json"

    save_csv(rows, csv_path)
    save_json(rows, json_path)

    summary = {

        "total_videos": total_videos,

        "datasets": dataset_counts,

        "csv": str(csv_path),

        "json": str(json_path),

    }

    return summary