"""
Common utility functions used throughout the ST-GCN++ preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict, List, Union
import json
import csv

import yaml
from rich.console import Console

console = Console()


# =============================================================================
# Configuration
# =============================================================================

def load_yaml(config_path: Union[str, Path]) -> Dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML file.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found:\n{config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Filesystem
# =============================================================================

def ensure_dir(directory: Union[str, Path]) -> None:
    """
    Create a directory if it does not already exist.
    """

    Path(directory).mkdir(parents=True, exist_ok=True)


def list_video_files(
    root_dir: Union[str, Path],
    extensions: List[str]
) -> List[Path]:
    """
    Recursively collect all supported video files.

    Parameters
    ----------
    root_dir : str | Path
        Dataset root directory.

    extensions : list
        Supported extensions from datasets.yaml

    Returns
    -------
    List[Path]
        Sorted list of video paths.
    """

    root_dir = Path(root_dir)

    videos = []

    extensions = {ext.lower() for ext in extensions}

    for file in root_dir.rglob("*"):

        if file.is_file() and file.suffix.lower() in extensions:
            videos.append(file)

    videos.sort()

    return videos


# =============================================================================
# Save utilities
# =============================================================================

def save_json(data: Dict, output_path: Union[str, Path]) -> None:
    """
    Save dictionary as JSON.
    """

    output_path = Path(output_path)

    ensure_dir(output_path.parent)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def save_csv(rows: List[Dict], output_path: Union[str, Path]) -> None:
    """
    Save list of dictionaries as CSV.
    """

    if len(rows) == 0:
        return

    output_path = Path(output_path)

    ensure_dir(output_path.parent)

    with open(output_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Logging
# =============================================================================

def info(message: str) -> None:
    console.print(f"[cyan][INFO][/cyan] {message}")


def success(message: str) -> None:
    console.print(f"[green][SUCCESS][/green] {message}")


def warning(message: str) -> None:
    console.print(f"[yellow][WARNING][/yellow] {message}")


def error(message: str) -> None:
    console.print(f"[red][ERROR][/red] {message}")