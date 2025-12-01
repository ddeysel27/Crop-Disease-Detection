from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE  

from utils.species_mapping import (
    DISEASE_MODEL_PATHS,
    DISEASE_LABELS,
    SPECIES_LIST,
)

PathLike = Union[str, Path]


# ============================================================
# VIT BASE BUILDER (matches your training architecture)
# ============================================================
def _build_vit_base(num_classes: int):
    """
    Build a TorchVision ViT-B_16 with a custom classification head.
    """
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# GENERIC VIT WEIGHT LOADER
# ============================================================
def load_vit_model(weights_path: PathLike, num_classes: int):
    """
    Load ViT-B_16 checkpoint with the correct head size.
    """
    model = _build_vit_base(num_classes=num_classes)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# SPECIES CLASSIFIER LOADER
# ============================================================
def load_species_model(device=DEVICE):
    """
    Load the species ViT classifier (16 classes).
    """
    weights = Path("models/species_classifier_vit.pth")
    num_classes = len(SPECIES_LIST)

    model = load_vit_model(weights, num_classes=num_classes)
    model.to(device)
    return model


# ============================================================
# DISEASE MODEL LOADER (Cassava / Rice / PlantVillage)
# ============================================================
def load_disease_model(model_path: PathLike, device=DEVICE):
    """
    Load the correct disease model using the clean mapping:

        DISEASE_MODEL_PATHS = {
            "Cassava": "...",
            "Rice": "...",
            "PlantVillage": "..."
        }

        DISEASE_LABELS = {
            "Cassava": [...],
            "Rice": [...],
            "PlantVillage": [...]
        }

    No filename hacks. We directly match the path.
    """
    model_path = str(model_path)

    # Find the correct model key (Cassava, Rice, PlantVillage)
    model_key = None
    for key, path in DISEASE_MODEL_PATHS.items():
        if str(path) == model_path:
            model_key = key
            break

    # Should never happen, but fallback:
    if model_key is None:
        model_key = "PlantVillage"

    # Number of disease classes for this model
    num_classes = len(DISEASE_LABELS[model_key])

    # Load vital checkpoint with correct head size
    model = load_vit_model(model_path, num_classes)
    model.to(device)
    return model
