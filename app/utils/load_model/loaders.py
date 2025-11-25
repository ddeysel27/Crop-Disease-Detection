from pathlib import Path
from typing import Union
import os

import torch
from ultralytics import YOLO
from torchvision import models

from config import DEVICE


PathLike = Union[str, Path]


def load_yolo_seg_model(weights_path: PathLike) -> YOLO:
    """
    Loads a YOLO segmentation model (e.g. yolov8n-seg) from given .pt.
    """
    weights_path = str(weights_path)
    model = YOLO(weights_path)
    return model


def _build_vit_base(num_classes: int = None):
    """
    Builds a ViT-Base model. Adjust to match how you trained your models.
    If you used timm, replace this with timm.create_model(...).
    """
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    if num_classes is not None:
        in_features = vit.heads.head.in_features
        vit.heads.head = torch.nn.Linear(in_features, num_classes)
    return vit


def load_vit_model(weights_path: PathLike, num_classes: int = None) -> torch.nn.Module:
    """
    Loads a ViT model and applies pretrained weights.
    """
    weights_path = Path(weights_path)
    model = _build_vit_base(num_classes=num_classes)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

from utils.species_mapping import SPECIES_NUM_CLASSES, DISEASE_LABELS


def load_species_model(device="cpu"):
    from utils.species_mapping import SPECIES_LIST

    weights = Path("models/species_classifier_vit.pth")
    num_classes = len(SPECIES_LIST)  # 16 classes

    model = load_vit_model(weights, num_classes=num_classes)
    model.to(device)
    return model




def load_disease_model(weights_path: PathLike, device="cpu"):
    """
    Loads a disease classifier for a given species (Cassava, Rice, PlantVillage).
    """
    from utils.species_mapping import DISEASE_LABELS, SPECIES_TO_MODEL

    weights_path = str(weights_path)

    # infer model key (Cassava / Rice / PlantVillage)
    model_key = None
    for key, path in SPECIES_TO_MODEL.items():
        if os.path.basename(path) in weights_path:
            model_key = key
            break

    if model_key is None:
        model_key = "PlantVillage"   # fallback

    num_classes = len(DISEASE_LABELS[model_key])

    model = load_vit_model(weights_path, num_classes=num_classes)
    model.to(device)
    return model


