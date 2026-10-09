"""
SUSHRUTA Inference Engine Module.

Provides reusable MedicalImageSegmenter class for loading trained U-Net checkpoint once at application startup,
selecting optimal compute device (CUDA / CPU), and running high-speed 2D FLAIR MRI segmentation inference.
"""

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

# Ensure project root is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D, count_parameters


class MedicalImageSegmenter:
    """
    Production-style U-Net Inference Engine for 2D Brain MRI Segmentation.

    Loads model checkpoint once during initialization, handles hardware device allocation (CUDA/CPU),
    and executes optimized forward inference passes.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initializes MedicalImageSegmenter engine.

        Parameters:
            checkpoint_path (Optional[str]): Path to .pt model checkpoint file.
            device (Optional[str]): Target device ('cuda' or 'cpu'). Auto-selects if None.
            threshold (float): Decision threshold for binary tumor mask conversion (default: 0.5).
        """
        if checkpoint_path is None:
            checkpoint_path = os.environ.get(
                "MODEL_CHECKPOINT_PATH",
                os.path.join(project_root, "results", "models", "baseline_01_clean_v2_best.pt")
            )

        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.threshold = threshold

        # Hardware Device Selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFERENCE ENGINE] Initializing MedicalImageSegmenter on device: {self.device}", flush=True)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"CRITICAL ERROR: U-Net Checkpoint not found at path: {self.checkpoint_path}")

        # Instantiate UNet2D model architecture
        self.model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(self.device)
        self.num_params = count_parameters(self.model)

        # Load state dict
        print(f"[INFERENCE ENGINE] Loading model weights from: {self.checkpoint_path}", flush=True)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.checkpoint_epoch = checkpoint.get("epoch", None)
            self.val_tumor_dice = checkpoint.get("val_tumor_dice", None)
        else:
            self.model.load_state_dict(checkpoint)
            self.checkpoint_epoch = None
            self.val_tumor_dice = None

        self.model.eval()
        print(f"[INFERENCE ENGINE] Checkpoint loaded successfully ({self.num_params:,} parameters, eval mode).", flush=True)

        # Warmup GPU if CUDA
        if self.device.type == "cuda":
            dummy = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                _ = self.model(dummy)
            torch.cuda.synchronize()
            print("[INFERENCE ENGINE] CUDA Warmup completed successfully.", flush=True)

    def segment(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes U-Net segmentation forward pass on preprocessed MRI input tensor.

        Parameters:
            input_tensor (torch.Tensor): Tensor of shape (1, 1, 256, 256), float32.

        Returns:
            prob_map (np.ndarray): 2D float array (256, 256) of continuous probabilities in [0.0, 1.0].
            binary_mask (np.ndarray): 2D uint8 array (256, 256) of binary tumor mask {0, 1}.
            inference_time_ms (float): Pure GPU/CPU forward pass latency in milliseconds.
        """
        assert input_tensor.shape == (1, 1, 256, 256), f"Expected input tensor shape (1, 1, 256, 256), got {input_tensor.shape}"

        input_tensor = input_tensor.to(self.device, non_blocking=True)

        # Precise execution latency measurement
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.inference_mode():
                logits = self.model(input_tensor)
                probs = torch.sigmoid(logits)
            end_event.record()
            torch.cuda.synchronize()
            inference_time_ms = start_event.elapsed_time(end_event)
        else:
            t_start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(input_tensor)
                probs = torch.sigmoid(logits)
            t_end = time.perf_counter()
            inference_time_ms = (t_end - t_start) * 1000.0

        prob_map = probs.squeeze().cpu().numpy().astype(np.float32)
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)

        return prob_map, binary_mask, round(float(inference_time_ms), 3)

    def get_info(self) -> Dict[str, Any]:
        """Returns model status metadata for health check."""
        return {
            "model_name": "UNet2D",
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "trainable_parameters": self.num_params,
            "checkpoint_epoch": self.checkpoint_epoch,
            "val_tumor_dice_stored": self.val_tumor_dice,
            "threshold": self.threshold
        }
