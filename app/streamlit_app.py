import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

from src.config import MODELS_DIR, DEVICE
from src.models.loaders import (
    load_yolo_seg_model,
    load_vit_model
)
from src.pipeline.inference import (
    apply_mask_and_crop,
    preprocess_leaf,
)

# ---- PAGE TITLE ----
st.set_page_config(page_title="Crop Disease Detection", layout="wide")
st.title("🌿 Crop Disease Detection — Streamlit App")


# ---- LOAD MODELS (cached for speed) ----
@st.cache_resource
def load_all_models():
    yolo = load_yolo_seg_model(MODELS_DIR / "yolo_plantdoc_seg.pt").to(DEVICE)
    species = load_vit_model(MODELS_DIR / "species_classifier_vit.pth").to(DEVICE)
    cassava = load_vit_model(MODELS_DIR / "cassava_best.pth").to(DEVICE)
    rice = load_vit_model(MODELS_DIR / "rice_best.pth").to(DEVICE)
    plantv = load_vit_model(MODELS_DIR / "plant_village_best.pth").to(DEVICE)

    disease_models = {
        "cassava": cassava,
        "rice": rice,
        "plantvillage": plantv
    }

    species_idx2name = {
        0: "cassava",
        1: "rice",
        2: "plantvillage"
    }

    DISEASE_LABEL_MAP = {
        "cassava": [
            "cassava_bacterial_blight",
            "cassava_brown_streak",
            "cassava_green_mottle",
            "cassava_healthy",
        ],
        "rice": [
            "rice_bacterial_leaf_blight",
            "rice_brown_spot",
            "rice_leaf_smut",
        ],
        "plantvillage": [
            "tomato_bacterial_spot",
            "tomato_early_blight",
            "tomato_healthy",
            # add your full list
        ]
    }

    return yolo, species, disease_models, species_idx2name, DISEASE_LABEL_MAP


yolo_model, species_model, disease_models, species_idx2name, DISEASE_LABEL_MAP = load_all_models()


# ---- FILE UPLOAD ----
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original Image")
        st.image(uploaded)

    # Convert uploaded image → BGR (for YOLO)
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ---- YOLO Segmentation ----
    leaf_rgb = apply_mask_and_crop(yolo_model, img_bgr)

    with col2:
        st.subheader("🍃 Segmented Leaf")
        st.image(leaf_rgb)

    # ---- Preprocess for ViT ----
    leaf_tensor = preprocess_leaf(leaf_rgb).unsqueeze(0).to(DEVICE)

    # ---- Species Prediction ----
    with torch.no_grad():
        species_logits = species_model(leaf_tensor)
        species_idx = torch.argmax(species_logits, dim=1).item()
        species_name = species_idx2name[species_idx]

    # ---- Disease Prediction ----
    disease_model = disease_models[species_name]

    with torch.no_grad():
        disease_logits = disease_model(leaf_tensor)
        disease_idx = torch.argmax(disease_logits, dim=1).item()
        disease_name = DISEASE_LABEL_MAP[species_name][disease_idx]

    # ---- RESULTS ----
    st.markdown("---")
    st.subheader("🔍 Results")
    st.write(f"**Predicted Species:** {species_name}")
    st.write(f"**Predicted Disease:** `{disease_name}`")
