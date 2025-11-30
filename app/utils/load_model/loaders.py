from pathlib import Path
from typing import Union
import os
import torch
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO

# Device comes from your config.py
from config import DEVICE  

PathLike = Union[str, Path]

# ============================
# VIT BASE BUILDER (TorchVision)
# ============================
def _build_vit_base(num_classes: int):
    """
    Build a TorchVision ViT-B_16 model with a custom classification head.
    This MUST match the architecture used during training.
    """
    model = models.vit_b_16(weights=None)   # Do NOT load pretrained since you trained your own
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ============================
# GENERIC VIT WEIGHT LOADER
# ============================
def load_vit_model(weights_path: PathLike, num_classes: int):
    """
    Load a ViT-B_16 model with the correct output size and apply checkpoint weights.
    """
    weights_path = Path(weights_path)

    model = _build_vit_base(num_classes=num_classes)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


# ============================
# IMPORT SPECIES MAPPING
# ============================
from utils.species_mapping import SPECIES_LIST, DISEASE_LABELS, SPECIES_TO_MODEL


# ============================
# LOAD SPECIES CLASSIFIER
# ============================
def load_species_model(device=DEVICE):
    """
    Load the 16-class species classifier (ViT-B16).
    """
    weights = Path("models/species_classifier_vit.pth")
    num_classes = len(SPECIES_LIST)  # always 16

    model = load_vit_model(weights, num_classes=num_classes)
    model.to(device)
    return model


# ============================
# LOAD DISEASE CLASSIFIER
# ============================
def load_disease_model(weights_path: PathLike, device=DEVICE):
    """
    Load Cassava / Rice / PlantVillage disease models.
    Output layer size is inferred from species_mapping.py.
    """
    weights_path = str(weights_path)

    # Determine model key (Cassava / Rice / PlantVillage)
    model_key = None
    for key, path in SPECIES_TO_MODEL.items():
        if os.path.basename(path) in weights_path:
            model_key = key
            break

    if model_key is None:
        model_key = "PlantVillage"  # fallback

    num_classes = len(DISEASE_LABELS[model_key])

    model = load_vit_model(weights_path, num_classes=num_classes)
    model.to(device)
    return model
