"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the ResNet18-based
Alzheimer's classifier.

Grad-CAM highlights which regions of the input MRI most influenced the model's
prediction, by backpropagating the gradient of the predicted class score into
the last convolutional layer and weighting the feature maps by those gradients.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017) — https://arxiv.org/abs/1610.02391
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Wraps a model + target layer and produces a class activation heatmap for a
    given input tensor.

    For this project's ResNet18, the target layer is `model.layer4` — the last
    convolutional block before global average pooling — since it has the
    highest-level spatial features while still retaining spatial resolution.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        """
        Args:
            input_tensor: a single preprocessed image, shape [1, C, H, W].
            class_idx: which class to explain. If None, uses the model's
                top predicted class.

        Returns:
            heatmap: numpy array of shape [H, W], values normalized to [0, 1]
            class_idx: the class index the heatmap explains
            probs: softmax probabilities over all classes (numpy array)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]

        heatmap = F.relu(weighted_activations).squeeze().cpu().numpy()

        # Normalize to [0, 1] for visualization; guard against a all-zero map
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, class_idx, probs

def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Resizes the (typically low-resolution) Grad-CAM heatmap up to the original
    image size and overlays it as a semi-transparent color map.

    Args:
        heatmap: 2D array in [0, 1], output of GradCAM.generate().
        original_image: the original PIL image (any size, will be used as the base).
        alpha: blend strength of the heatmap (0 = invisible, 1 = fully opaque).
        colormap: a matplotlib colormap name.

    Returns:
        A PIL Image with the heatmap overlaid on the original image.
    """
    import matplotlib

    original_image = original_image.convert("RGB")
    w, h = original_image.size

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img) / 255.0

    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap(colormap)

    colored_heatmap = cmap(heatmap_arr)[:, :, :3]  # drop alpha channel from colormap
    colored_heatmap = np.uint8(colored_heatmap * 255)

    original_arr = np.array(original_image)
    blended = (alpha * colored_heatmap + (1 - alpha) * original_arr).astype(np.uint8)

    return Image.fromarray(blended)