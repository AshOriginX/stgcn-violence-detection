"""
Model loading utilities for the ST-GCN++ preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict

import torch
from ultralytics import YOLO


class ModelFactory:
    """
    Factory class responsible for loading AI models.
    """

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Resolve computation device.
        """

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA unavailable. Falling back to CPU.")
            return "cpu"

        return device

    @staticmethod
    def load_yolo(config: Dict) -> YOLO:
        """
        Load a YOLO model.

        Parameters
        ----------
        config : dict
            Detector configuration.

        Returns
        -------
        ultralytics.YOLO
        """

        model_name = config["model"]

        print(f"\nLoading YOLO model: {model_name}")

        model = YOLO(model_name)

        device = ModelFactory._resolve_device(config["device"])

        model.to(device)

        print(f"Device : {device}")

        return model

    @staticmethod
    def load_mmpose(config: Dict):
        """
        Load an MMPose model.

        Parameters
        ----------
        config : dict
            Pose estimator configuration with keys:
            - config_file: str, path to model config file
            - checkpoint_file: str, path to model checkpoint file
            - device: str, computation device

        Returns
        -------
        mmpose model
        """

        config_file = config["config_file"]
        checkpoint_file = config["checkpoint_file"]

        print(f"\nLoading MMPose model: {config_file}")
        print(f"Checkpoint: {checkpoint_file}")

        from mmpose.apis import init_model

        device = ModelFactory._resolve_device(config["device"])

        model = init_model(config_file, checkpoint_file, device=device)

        print(f"Device : {device}")

        return model