"""
SUSHRUTA LayerCAM Explainable AI (XAI) Module.

Implements LayerCAM (Jiang et al., IEEE TPAMI 2021) for 2D U-Net medical image segmentation.
Calculates element-wise spatial weights from backpropagated target gradients:
    w_{i,j}^k = max( dY / dA_{i,j}^k, 0 )
    L_{i,j}   = max( sum_k w_{i,j}^k * A_{i,j}^k, 0 )
Normalized to [0.0, 1.0] and resized to 256x256.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn


class LayerCAMGenerator:
    """
    LayerCAM Explainability Generator for U-Net Semantic Segmentation.

    Safely hooks target convolutional layer, executes backward pass on segmentation target score Y,
    calculates spatial attribution map, and removes hooks cleanly to prevent GPU memory leaks.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initializes LayerCAM Generator.

        Parameters:
            model (nn.Module): Active PyTorch UNet2D model instance.
            target_layer_name (Optional[str]): Target convolutional layer name. Auto-detects if None.
            device (Optional[torch.device]): Hardware compute device.
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

        # Locate target layer: Default to up4.conv.double_conv[3] (final decoder conv layer before outc)
        if target_layer_name is None:
            target_layer_name = "up4.conv.double_conv.3"

        self.target_layer_name = target_layer_name
        self.target_layer = self._find_target_layer(target_layer_name)

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    def _find_target_layer(self, layer_name: str) -> nn.Module:
        """Helper to resolve target module from model by dot-separated name path."""
        parts = layer_name.split(".")
        curr = self.model
        for p in parts:
            if p.isdigit():
                curr = curr[int(p)]
            elif hasattr(curr, p):
                curr = getattr(curr, p)
            else:
                # Fallback to up4.conv.double_conv[3] if exact path fails
                if hasattr(self.model, "up4"):
                    curr = self.model.up4.conv.double_conv[3]
                    layer_name = "up4.conv.double_conv.3"
                    break
                else:
                    raise AttributeError(f"Target layer '{layer_name}' not found in model architecture.")
        return curr

    def _forward_hook(self, module: nn.Module, input_tensor: Tuple[torch.Tensor], output_tensor: torch.Tensor):
        self._activations = output_tensor.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            self._gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        binary_mask: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """
        Generates LayerCAM attribution heatmap for an input MRI tensor.

        Parameters:
            input_tensor (torch.Tensor): Tensor of shape (1, 1, 256, 256), float32.
            binary_mask (np.ndarray): Binary numpy array (256, 256) of predicted tumor region {0, 1}.
            threshold (float): Decision threshold.

        Returns:
            cam_norm (np.ndarray): 2D float numpy array (256, 256) normalized to [0.0, 1.0].
            explainability_meta (Dict[str, Any]): Metadata dict explaining method, target layer, and target definition.
            layercam_time_ms (float): Execution latency of LayerCAM in milliseconds.
        """
        assert input_tensor.shape == (1, 1, 256, 256), f"Expected input tensor (1, 1, 256, 256), got {input_tensor.shape}"

        t_start = time.perf_counter()

        self._activations = None
        self._gradients = None

        # Register PyTorch forward & backward hooks
        fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

        input_tensor = input_tensor.to(self.device, non_blocking=True).requires_grad_(True)

        try:
            # Enable gradient calculation for LayerCAM backward pass
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                logits = self.model(input_tensor)

                tumor_pixel_count = int(np.sum(binary_mask > 0))
                if tumor_pixel_count > 0:
                    # Target Y = Sum of raw un-normalized logits over predicted tumor region
                    mask_tensor = torch.from_numpy(binary_mask).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    target_score = (logits * mask_tensor).sum()
                    target_def = f"Sum of U-Net raw logit activations over predicted tumor pixels ({tumor_pixel_count} px)"
                else:
                    # Target Y = Maximum logit activation across slice if no tumor predicted
                    target_score = logits.max()
                    target_def = "Maximum logit activation across slice (no tumor pixels predicted)"

                if torch.isnan(target_score) or torch.isinf(target_score):
                    raise ValueError("Target score for LayerCAM backpropagation is non-finite (NaN or Inf).")

                # Backpropagate target score Y to target layer feature maps A
                target_score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Failed to capture activations or gradients during LayerCAM backward pass.")

            A = self._activations  # (1, K, H, W)
            G = self._gradients    # (1, K, H, W)

            # -----------------------------------------------------------------
            # LayerCAM Calculation (Jiang et al., IEEE TPAMI 2021)
            # w_{i,j}^k = max( dY / dA_{i,j}^k, 0 )  [ReLU on gradients]
            # L_{i,j}   = max( sum_k w_{i,j}^k * A_{i,j}^k, 0 ) [ReLU on weighted sum]
            # -----------------------------------------------------------------
            weights = torch.clamp(G, min=0.0)
            cam_tensor = torch.clamp((weights * A).sum(dim=1, keepdim=True), min=0.0)

            cam_np = cam_tensor.squeeze().cpu().numpy().astype(np.float32)
            cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=0.0, neginf=0.0)

        finally:
            # Safely remove handles to prevent hook accumulation and GPU memory leaks
            fwd_handle.remove()
            bwd_handle.remove()
            self.model.zero_grad(set_to_none=True)
            self._activations = None
            self._gradients = None

        # Resize CAM to (256, 256) if target layer resolution differs
        if cam_np.shape != (256, 256):
            cam_np = cv2.resize(cam_np, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Min-Max Normalize CAM to range [0.0, 1.0]
        min_v, max_v = float(np.min(cam_np)), float(np.max(cam_np))
        if max_v > min_v + 1e-8:
            cam_norm = (cam_np - min_v) / (max_v - min_v)
        else:
            cam_norm = np.zeros((256, 256), dtype=np.float32)

        t_end = time.perf_counter()
        layercam_time_ms = round((t_end - t_start) * 1000.0, 2)

        explainability_meta = {
            "method": "LayerCAM",
            "target_layer": self.target_layer_name,
            "target_definition": target_def,
            "interpretation": (
                "Regions with stronger activation (higher values) indicate spatial locations "
                "that contributed most strongly to the model's tumor segmentation prediction."
            ),
            "layercam_time_ms": layercam_time_ms
        }

        return cam_norm, explainability_meta, layercam_time_ms
