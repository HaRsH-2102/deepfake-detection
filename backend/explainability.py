from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

from backend.predictor import get_model, preprocess_image
from backend.utils import rgb_to_base64_png


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_: torch.nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(
            _: torch.nn.Module,
            __: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor.float())
        if output.ndim > 1 and output.shape[1] > 1 and class_idx is not None:
            target = output[:, class_idx].sum()
        else:
            target = output[:, 0].sum() if output.ndim > 1 else output.sum()
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        denominator = cam.max() if cam.max() > 0 else 1.0
        cam = cam / denominator
        return np.uint8(cam * 255)


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    candidate: torch.nn.Module | None = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            candidate = module
    if candidate is None:
        raise RuntimeError("Could not locate a convolutional layer for Grad-CAM.")
    return candidate


def run_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
    original_image: np.ndarray | None = None,
) -> str:
    layer = target_layer or find_target_layer(model)
    gradcam = GradCAM(model, layer)
    heatmap = gradcam.generate(image_tensor)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if original_image is None:
        original = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        original = np.uint8(np.clip(original * 255.0, 0, 255))
    else:
        original = cv2.resize(original_image, (image_tensor.shape[-1], image_tensor.shape[-2]))

    overlay = cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)
    return rgb_to_base64_png(overlay)


def run_lime(
    model: torch.nn.Module,
    face_image_array: np.ndarray,
    num_samples: int = 500,
    num_features: int = 5,
) -> str:
    face_rgb = cv2.resize(face_image_array, (128, 128))
    explainer = LimeImageExplainer()
    runtime_device = next(model.parameters()).device

    def predict_fn(images: list[np.ndarray]) -> np.ndarray:
        batch = torch.cat([preprocess_image(np.uint8(image)) for image in images], dim=0)
        batch = batch.to(runtime_device)
        with torch.no_grad():
            logits = model(batch.float())
            probabilities = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
        return np.stack([1.0 - probabilities, probabilities], axis=1)

    explanation = explainer.explain_instance(
        face_rgb,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
    )

    label_index = 1 if 1 in explanation.top_labels else explanation.top_labels[0]
    _, mask = explanation.get_image_and_mask(
        label_index,
        positive_only=False,
        num_features=num_features,
        hide_rest=False,
    )

    overlay = face_rgb.copy().astype(np.float32)
    green = np.array([34, 197, 94], dtype=np.float32)
    red = np.array([239, 68, 68], dtype=np.float32)

    overlay[mask > 0] = overlay[mask > 0] * 0.45 + green * 0.55
    overlay[mask < 0] = overlay[mask < 0] * 0.45 + red * 0.55
    outlined = mark_boundaries(np.clip(overlay / 255.0, 0, 1), mask, color=(1, 1, 1))
    return rgb_to_base64_png(np.uint8(outlined * 255))


def run_explainability(model: torch.nn.Module | None, face_array: np.ndarray) -> dict[str, str]:
    runtime_model = model or get_model()
    image_tensor = preprocess_image(face_array)
    return {
        "gradcam_image": run_gradcam(runtime_model, image_tensor, original_image=face_array),
        "lime_image": run_lime(runtime_model, face_array),
    }
