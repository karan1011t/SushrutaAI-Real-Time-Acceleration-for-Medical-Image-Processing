"""
SUSHRUTA — Binary Tumor Classifier LayerCAM XAI & Spatial Location Generator.

Provides LayerCAM gradient-based feature attribution for BinaryTumorClassifier:
- Hooks final convolutional layer (features[12]) of BinaryTumorClassifier.
- Computes LayerCAM element-wise weighted activation heatmap L ∈ [0, 1]^(256x256).
- Extracts highest-contribution spatial region bounding box and centroid (cx, cy).
- Derives image-relative spatial location (e.g. center-right, upper-left).
- Generates dynamic 1-2 sentence natural language explanation derived purely from prediction data.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any, Tuple


class ClassifierLayerCAMGenerator:
    """
    LayerCAM Explainability Generator for BinaryTumorClassifier.
    Target Layer: features[12] (nn.Conv2d(128, 256, kernel_size=3, padding=1))
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        # Target layer is last Conv2d layer in features (features[12])
        self.target_layer = model.features[12]
        self.target_layer_name = "features.12"
        self.activations = None
        self.gradients = None

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        decision_threshold: float = 0.5
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """
        Generates Classifier LayerCAM heatmap for an input MRI tensor.

        Returns:
            cam_norm (np.ndarray): 2D float array (256, 256) in [0.0, 1.0].
            tumor_probability (float): Sigmoid probability of tumor presence [0.0, 1.0].
            tumor_logit (float): Raw output logit score.
            location_info (Dict[str, Any]): Spatial location metadata {location, centroid, bounding_box}.
        """
        self.model.eval()
        self.model.zero_grad()

        h_act = self.target_layer.register_forward_hook(self._save_activation)
        h_grad = self.target_layer.register_full_backward_hook(self._save_gradient)

        try:
            tensor_in = input_tensor.to(self.device).requires_grad_(True)
            logit = self.model(tensor_in)
            prob = float(torch.sigmoid(logit).item())
            logit_val = float(logit.item())

            # Backpropagate target class score
            target_score = logit[0, 0]
            target_score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                # Fallback to zero map if hooks fail
                cam_norm = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
                loc_info = self._derive_location(cam_norm, is_detected=False)
                return cam_norm, prob, logit_val, loc_info

            act = self.activations[0]  # (256, H, W)
            grad = self.gradients[0]    # (256, H, W)

            # LayerCAM element-wise spatial weighting: max(0, grad) * act
            spatial_weights = torch.clamp(grad, min=0.0) * act
            cam = torch.sum(spatial_weights, dim=0)  # (H, W)
            cam = torch.clamp(cam, min=0.0)

            cam_np = cam.cpu().numpy()
            H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]
            cam_resized = cv2.resize(cam_np, (W_in, H_in), interpolation=cv2.INTER_LINEAR)

            c_min, c_max = float(cam_resized.min()), float(cam_resized.max())
            is_detected = bool(prob >= decision_threshold)

            if is_detected:
                # Positive classification: min-max normalize to [0.0, 1.0] to highlight focal region
                if c_max > c_min:
                    cam_norm = (cam_resized - c_min) / (c_max - c_min)
                else:
                    cam_norm = np.zeros_like(cam_resized, dtype=np.float32)
            else:
                # Sub-threshold classification:
                # If prob < 1e-4 or flat, suppress completely to prevent fake noise lesions
                # Otherwise, confidence-scale by (prob / decision_threshold)
                if c_max > c_min and prob >= 1e-4:
                    cam_norm = ((cam_resized - c_min) / (c_max - c_min)) * (prob / decision_threshold)
                else:
                    cam_norm = np.zeros_like(cam_resized, dtype=np.float32)

            location_info = self._derive_location(cam_norm, is_detected=is_detected)

            return cam_norm, prob, logit_val, location_info
        finally:
            h_act.remove()
            h_grad.remove()
            self.activations = None
            self.gradients = None

    def _derive_location(self, cam_norm: np.ndarray, is_detected: bool = True) -> Dict[str, Any]:
        """
        Derives image-relative spatial location (e.g. center-right, upper-left) from LayerCAM.
        For sub-threshold / non-detected slices or zero-activation maps, returns clean non-focal metadata.
        """
        if not is_detected or float(cam_norm.max()) <= 1e-4:
            return {
                'location': 'none',
                'centroid': {'x': 0.0, 'y': 0.0},
                'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0}
            }

        H, W = cam_norm.shape
        max_val = float(cam_norm.max())
        thresh_val = 0.5 * max_val
        high_act = (cam_norm >= thresh_val).astype(np.uint8)

        if np.sum(high_act) == 0:
            return {
                'location': 'none',
                'centroid': {'x': 0.0, 'y': 0.0},
                'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0}
            }

        y_indices, x_indices = np.where(high_act > 0)
        cx = float(np.mean(x_indices))
        cy = float(np.mean(y_indices))

        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

        # Horizontal alignment
        if cx < W / 3.0:
            x_label = 'left'
        elif cx > 2.0 * W / 3.0:
            x_label = 'right'
        else:
            x_label = 'center'

        # Vertical alignment
        if cy < H / 3.0:
            y_label = 'upper'
        elif cy > 2.0 * H / 3.0:
            y_label = 'lower'
        else:
            y_label = 'center'

        if x_label == 'center' and y_label == 'center':
            loc_str = 'center'
        elif x_label == 'center':
            loc_str = f'{y_label}-center'
        elif y_label == 'center':
            loc_str = f'center-{x_label}'
        else:
            loc_str = f'{y_label}-{x_label}'

        return {
            'location': loc_str,
            'centroid': {'x': round(cx, 2), 'y': round(cy, 2)},
            'bounding_box': {
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x + 1,
                'height': max_y - min_y + 1
            }
        }


def generate_dynamic_explanation(
    tumor_detected: bool,
    tumor_probability: float,
    decision_threshold: float,
    location_str: str
) -> str:
    """
    Generates dynamic 1-2 sentence human-readable explanation derived purely from prediction data.
    """
    prob_percent = round(tumor_probability * 100.0, 1)
    thresh_percent = round(decision_threshold * 100.0, 1)

    if tumor_detected:
        loc_desc = f"in the {location_str} portion of the MRI slice" if location_str and location_str != "none" else "across the slice"
        return (
            f"Tumor indicated by auxiliary screening model with {prob_percent}% probability (above the {thresh_percent}% decision threshold). "
            f"The classification decision was primarily influenced by the highlighted region {loc_desc}."
        )
    else:
        return (
            f"No tumor indicated by auxiliary screening model. The assigned tumor probability is {prob_percent}%, "
            f"which is below the {thresh_percent}% decision threshold. "
            f"No focal tumor attribution is confirmed; any visual heatmap represents sub-threshold feature traces."
        )
