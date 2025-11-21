from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.config import DEVICE


def apply_mask_and_crop(yolo_model, img_bgr: np.ndarray) -> np.ndarray:
    """
    Runs YOLO segmentation on a BGR image (OpenCV) and returns
    a cropped RGB leaf patch.
    """
    # YOLO expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model(img_rgb)[0]

    if results.masks is None or len(results.masks.data) == 0:
        # If no mask, just return the center-cropped image
        h, w, _ = img_rgb.shape
        size = min(h, w)
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        crop = img_rgb[start_y:start_y+size, start_x:start_x+size]
        return crop

    mask = results.masks.data[0].cpu().numpy()  # [H, W]
    mask = (mask > 0.5).astype(np.uint8)

    # Apply mask
    leaf = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # Find bounding box of mask
    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    leaf_crop = leaf[y_min:y_max, x_min:x_max]

    return leaf_crop


def preprocess_leaf(
    leaf_rgb: np.ndarray,
    img_size: int = 224,
) -> torch.Tensor:
    """
    Converts a cropped RGB leaf (H, W, 3) into a PyTorch tensor suitable
    for ViT.
    """
    img_pil = Image.fromarray(leaf_rgb)
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return tfms(img_pil)


def run_pipeline(
    img_path: Path,
    yolo_model,
    species_model,
    species_idx2name,
    disease_models: dict,
    disease_label_map: dict,
    img_size: int = 224,
) -> Tuple[str, str]:
    """
    Generic pipeline function used by scripts or apps.

    Returns:
        (species_pred, disease_pred)
    """
    # 1) Read and segment leaf
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    leaf_rgb = apply_mask_and_crop(yolo_model, img_bgr)

    # 2) Preprocess leaf for ViT
    leaf_tensor = preprocess_leaf(leaf_rgb, img_size=img_size).unsqueeze(0).to(DEVICE)

    # 3) Species prediction
    species_model.eval()
    with torch.no_grad():
        logits = species_model(leaf_tensor)
        species_idx = torch.argmax(logits, dim=1).item()

    species_name = species_idx2name[species_idx]

    if species_name not in disease_models:
        raise ValueError(f"No disease model configured for species '{species_name}'")

    # 4) Disease prediction
    disease_model = disease_models[species_name]
    disease_model.eval()
    with torch.no_grad():
        logits = disease_model(leaf_tensor)
        disease_idx = torch.argmax(logits, dim=1).item()

    disease_label_list = disease_label_map[species_name]
    disease_name = disease_label_list[disease_idx]

    return species_name, disease_name
