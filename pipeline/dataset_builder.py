"""
Dataset builder module for assembling PySKL training datasets.

This module provides production-quality dataset assembly with:
- Train/validation splitting
- Random seed for reproducibility
- PySKL-compatible dictionary construction
- PKL file serialization
- Comprehensive error handling and logging
- Type hints and detailed docstrings
- Single responsibility principle
"""

import logging
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Production-quality dataset builder for PySKL training datasets.

    This class assembles PySKL annotation dictionaries into train/validation
    splits and serializes them to PKL files for training.

    The builder does NOT run inference - it only assembles pre-computed
    annotations into the final dataset format.

    Attributes
    ----------
    output_dir : Path
        Directory where PKL files will be saved.
    annotations : List[Dict]
        List of PySKL annotation dictionaries.
    val_split : float
        Fraction of data to use for validation (0.0 to 1.0).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, output_dir: Path, val_split: float = 0.2, seed: int = 42):
        """
        Initialize the dataset builder.

        Parameters
        ----------
        output_dir : Path
            Directory to save PKL files. Will be created if it doesn't exist.
        val_split : float, optional
            Fraction of data to use for validation. Default is 0.2.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Raises
        ------
        ValueError
            If val_split is not between 0 and 1, or output_dir is invalid.
        OSError
            If output directory creation fails.
        """

        if val_split < 0 or val_split > 1:
            raise ValueError("val_split must be between 0 and 1")

        if output_dir is None:
            raise ValueError("output_dir cannot be None")

        self.output_dir = Path(output_dir)
        self._ensure_output_directory()

        self.annotations: List[Dict] = []
        self.val_split = val_split
        self.seed = seed

        logger.info(
            "Initialized DatasetBuilder with output_dir=%s, val_split=%.2f, seed=%d",
            self.output_dir,
            val_split,
            seed,
        )

    def _ensure_output_directory(self) -> None:
        """
        Ensure the output directory exists, creating it if necessary.

        Raises
        ------
        OSError
            If directory creation fails.
        """

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Output directory ready: %s", self.output_dir)

        except Exception as e:
            logger.error("Failed to create output directory %s: %s", self.output_dir, e)
            raise OSError(f"Failed to create output directory: {e}")

    def add(self, annotation: Dict) -> None:
        """
        Add a PySKL annotation dictionary to the dataset.

        Parameters
        ----------
        annotation : Dict
            PySKL annotation dictionary from PipelineExporter.

        Raises
        ------
        ValueError
            If annotation is invalid or None.
        """

        if annotation is None:
            raise ValueError("annotation cannot be None")

        required_keys = [
            "frame_dir",
            "label",
            "img_shape",
            "original_shape",
            "total_frames",
            "keypoint",
            "keypoint_score",
        ]

        for key in required_keys:
            if key not in annotation:
                raise ValueError(f"Annotation missing required key: {key}")

        self.annotations.append(annotation)
        logger.debug("Added annotation for video %s", annotation["frame_dir"])

    def build(self) -> None:
        """
        Build the dataset and write train/val PKL files.

        This method performs the following steps:
        1. Shuffle annotations using configured seed
        2. Split into train and validation sets
        3. Construct PySKL dictionary with split info
        4. Write train.pkl and val.pkl

        Raises
        ------
        ValueError
            If no annotations have been added.
        IOError
            If file writing fails.
        """

        if not self.annotations:
            raise ValueError("No annotations added. Call add() before build().")

        try:
            # Shuffle annotations with local Random instance
            rng = random.Random(self.seed)
            shuffled = self.annotations.copy()
            rng.shuffle(shuffled)

            logger.info("Shuffled %d annotations with seed %d", len(shuffled), self.seed)

            # Split into train and validation
            split_idx = int(len(shuffled) * (1 - self.val_split))
            train_annotations = shuffled[:split_idx]
            val_annotations = shuffled[split_idx:]

            logger.info(
                "Split dataset: %d train, %d val",
                len(train_annotations),
                len(val_annotations),
            )

            # Construct PySKL dictionaries
            train_dict = self._build_pyskl_dict(train_annotations, "train")
            val_dict = self._build_pyskl_dict(val_annotations, "val")

            # Write PKL files
            self._write_pkl(train_dict, "train.pkl")
            self._write_pkl(val_dict, "val.pkl")

            # Log class balance
            self._log_class_balance(train_annotations, "train")
            self._log_class_balance(val_annotations, "val")

            logger.info(
                "Dataset built successfully: %s/train.pkl, %s/val.pkl",
                self.output_dir,
                self.output_dir,
            )

        except Exception as e:
            logger.error("Failed to build dataset: %s", e)
            raise IOError(f"Dataset build failed: {e}")

    def _build_pyskl_dict(self, annotations: List[Dict], split_name: str) -> Dict:
        """
        Build a PySKL-compatible dictionary from annotations.

        Parameters
        ----------
        annotations : List[Dict]
            List of PySKL annotation dictionaries.
        split_name : str
            Name of the split (e.g., "train", "val").

        Returns
        -------
        Dict
            PySKL-compatible dictionary with split info and annotations.
        """

        return {
            "split": {split_name: [ann["frame_dir"] for ann in annotations]},
            "annotations": annotations,
        }

    def _write_pkl(self, data: Dict, filename: str) -> None:
        """
        Write data to a PKL file.

        Parameters
        ----------
        data : Dict
            Data to serialize.
        filename : str
            Name of the output file.

        Raises
        ------
        IOError
            If file writing fails.
        """

        output_path = self.output_dir / filename

        try:
            with open(output_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug("Wrote %s (%d annotations)", output_path, len(data["annotations"]))

        except Exception as e:
            logger.error("Failed to write %s: %s", output_path, e)
            raise IOError(f"Failed to write PKL file: {e}")

    def _log_class_balance(self, annotations: List[Dict], split_name: str) -> None:
        """
        Log class balance for a split.

        Parameters
        ----------
        annotations : List[Dict]
            List of PySKL annotation dictionaries.
        split_name : str
            Name of the split (e.g., "train", "val").
        """

        label_counts = Counter(ann["label"] for ann in annotations)

        logger.info("%s class balance:", split_name)
        for label, count in sorted(label_counts.items()):
            logger.info("  Label %d: %d", label, count)

    def close(self) -> None:
        """
        Clean up resources.

        This method performs any necessary cleanup operations.
        """

        logger.info("DatasetBuilder resources released.")
