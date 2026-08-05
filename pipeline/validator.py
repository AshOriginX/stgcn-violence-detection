"""
Dataset validation module for the ST-GCN++ preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from pipeline.utils import (
    load_yaml,
    save_csv,
    save_json,
    ensure_dir,
)
from pipeline.video import read_video_metadata


def _infer_label(path: Path) -> str:
    """Infer class label from directory names."""

    parts = {p.lower() for p in path.parts}

    if "fight" in parts or "violence" in parts:
        return "Fight"

    return "NonFight"


def validate_dataset(
    config_path: str = "configs/datasets.yaml"
) -> Dict:
    """
    Validate every configured dataset.

    Returns
    -------
    dict
        Validation summary.
    """

    config = load_yaml(config_path)

    report_dir = Path(config["outputs"]["reports"])
    ensure_dir(report_dir)

    report_rows: List[Dict] = []
    invalid_files: List[str] = []

    dataset_summary = {}

    total = 0
    valid = 0
    invalid = 0

    for dataset_key, dataset_info in config["datasets"].items():

        if dataset_key == "unified":
            continue

        dataset_name = dataset_info["name"]
        dataset_path = Path(dataset_info["path"])

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{dataset_path}"
            )

        video_files = []

        for ext in config["video_extensions"]:
            video_files.extend(dataset_path.rglob(f"*{ext}"))

        video_files = sorted(video_files)

        dataset_valid = 0
        dataset_invalid = 0

        print(f"\nScanning {dataset_name}...")

        for video in tqdm(video_files):

            metadata = read_video_metadata(video)

            metadata["dataset"] = dataset_name
            metadata["label"] = _infer_label(video)

            report_rows.append(metadata)

            total += 1

            if metadata["readable"]:
                valid += 1
                dataset_valid += 1
            else:
                invalid += 1
                dataset_invalid += 1
                invalid_files.append(str(video))

        dataset_summary[dataset_name] = {
            "total": len(video_files),
            "valid": dataset_valid,
            "invalid": dataset_invalid,
        }

    save_csv(
        report_rows,
        report_dir / "dataset_report.csv"
    )

    with open(report_dir / "invalid_videos.txt", "w") as f:
        for file in invalid_files:
            f.write(file + "\n")

    summary = {
        "total_videos": total,
        "valid_videos": valid,
        "invalid_videos": invalid,
        "datasets": dataset_summary,
    }

    save_json(
        summary,
        report_dir / "summary.json"
    )

    return summary