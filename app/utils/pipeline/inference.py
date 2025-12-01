import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.explainability.gradcam_pp import GradCAMPP
from utils.model_utils import clean_label
from utils.species_mapping import (
    SPECIES_LIST,
    SPECIES_TO_MODEL_KEY,
    DISEASE_MODEL_PATHS,
    DISEASE_LABELS,
)
from utils.load_model.loaders import load_species_model, load_disease_model
from utils.detection.leaf_detector import LeafDetector
from utils.ood.ood_detector import OODDetector


class InferencePipeline:
    def __init__(self, device: str = "cpu"):
        self.device = device

        # 1) Species classifier
        self.species_model = load_species_model(device=self.device)

        # 2) Disease classifiers: 3 keys total
        #    "Cassava", "Rice", "PlantVillage"
        self.disease_models = {
            key: load_disease_model(path, device=self.device)
            for key, path in DISEASE_MODEL_PATHS.items()
        }

        # 3) YOLO leaf detector
        self.detector = LeafDetector("models/yolo_plantdoc_detect.pt")

        # 4) OOD detector for species logits
        self.ood = OODDetector()

        # 5) Base transforms
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    # -------------------------------------------------
    # Helper transforms for TTA
    # -------------------------------------------------
    def tta_transforms(self, img: Image.Image):
        """Return a list of augmented images for TTA."""
        return [
            img,
            TF.hflip(img),
            TF.rotate(img, 10),
            TF.rotate(img, -10),
            TF.adjust_brightness(img, 1.2),
            TF.adjust_contrast(img, 1.2),
        ]

    def tta_predict(self, model, img: Image.Image):
        """Run the model on several augmented versions and average outputs."""
        model.eval()
        preds = []

        for augmented in self.tta_transforms(img):
            tensor = self.preprocess(augmented)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

        preds = np.array(preds)           # [T, 1, C]
        mean_pred = preds.mean(axis=0)[0] # [C]
        return mean_pred

    def tta_uncertainty(self, model, img: Image.Image):
        preds = []

        for augmented in self.tta_transforms(img):
            tensor = self.preprocess(augmented)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

        preds = np.array(preds)[:, 0, :]  # [T, C]
        uncertainty = preds.std(axis=0).mean()
        return float(uncertainty)

    # -------------------------------------------------
    # Core utilities
    # -------------------------------------------------
    def preprocess(self, img: Image.Image):
        return self.tfms(img).unsqueeze(0).to(self.device)

    def predict_species(self, img_tensor):
        self.species_model.eval()
        with torch.no_grad():
            logits = self.species_model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return idx.item(), conf.item()

    def predict_disease(self, disease_key: str, img_tensor):
        """
        disease_key ∈ {'Cassava', 'Rice', 'PlantVillage'}
        Simple helper if you ever want a direct call.
        """
        model = self.disease_models[disease_key]
        model.eval()

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        return idx.item(), conf.item()

    def mc_dropout_predict(self, model, img_tensor, passes: int = 15):
        """
        Monte Carlo dropout for uncertainty.
        """
        model.train()  # enable dropout layers

        preds = []
        with torch.no_grad():
            for _ in range(passes):
                logits = model(img_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=1)
                preds.append(probs)

        preds = torch.stack(preds)       # [passes, 1, C]
        mean_pred = preds.mean(dim=0)    # [1, C]

        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum().item()
        return mean_pred, entropy

    # -------------------------------------------------
    # MAIN PREDICT (used in 02_Upload_and_Classify)
    # -------------------------------------------------
    def predict(self, image: Image.Image) -> dict:
        # 0) Leaf detection (YOLO)
        cropped_image, boxed = self.detector.detect_leaf(image)
        if cropped_image is None:
            return {
                "error": "No leaf detected. Please upload a clear photo of a plant leaf."
            }

        # 1) Preprocess
        img_tensor = self.preprocess(cropped_image)

        # 2) OOD check (species logits)
        with torch.no_grad():
            species_logits = self.species_model(img_tensor)

        if self.ood.is_ood(species_logits)["is_ood"]:
            return {
                "error": "❗ The image does not look like a valid plant leaf or is out of distribution."
            }

        # 3) Species prediction
        probs = torch.softmax(species_logits, dim=1)
        species_conf, species_idx = probs.max(dim=1)
        species_idx = species_idx.item()
        species_conf = species_conf.item()

        species_name = SPECIES_LIST[species_idx]

        # 4) Choose disease model key via mapping
        disease_key = SPECIES_TO_MODEL_KEY.get(species_name, "PlantVillage")
        model = self.disease_models[disease_key]

        # 5) MC-dropout prediction
        mc_mean_pred, mc_uncertainty = self.mc_dropout_predict(model, img_tensor)
        mc_mean_pred = mc_mean_pred.squeeze()
        if mc_mean_pred.ndim != 1:
            mc_mean_pred = mc_mean_pred.view(-1)

        disease_idx = int(mc_mean_pred.argmax())
        disease_conf_mc = float(mc_mean_pred[disease_idx])

        # 6) TTA prediction
        tta_mean_pred = self.tta_predict(model, cropped_image)
        tta_uncertainty = self.tta_uncertainty(model, cropped_image)

        if tta_mean_pred.ndim != 1:
            tta_mean_pred = tta_mean_pred.reshape(-1)

        disease_conf_tta = float(tta_mean_pred[disease_idx])

        # 7) Fuse MC + TTA confidences & uncertainties
        fused_conf = (disease_conf_mc + disease_conf_tta) / 2.0
        fused_uncertainty = (mc_uncertainty + tta_uncertainty) / 2.0

        # 8) Final disease label (from correct label list)
        raw_label = DISEASE_LABELS[disease_key][disease_idx]
        disease_label = clean_label(raw_label)

        # 9) GradCAM++ heatmap
        grad = GradCAMPP(self.disease_models[disease_key])
        heatmap_mask = grad(img_tensor, disease_idx)
        heatmap_img = GradCAMPP.overlay(cropped_image, heatmap_mask)

        # 10) Save latest result for LLM page (best-effort, ignore failures)
        try:
            with open("latest_result.txt", "w") as f:
                f.write(f"Species: {species_name}\n")
                f.write(f"Disease: {disease_label}\n")
                f.write(f"Species_Confidence: {species_conf:.4f}\n")
                f.write(f"Disease_Confidence_Fused: {fused_conf:.4f}\n")
                f.write(f"Uncertainty_Fused: {fused_uncertainty:.4f}\n")
        except Exception:
            pass

        # 11) Return all info to Streamlit
        return {
            "species": species_name,
            "species_conf": float(species_conf),
            "disease": disease_label,
            "disease_conf": float(fused_conf),
            "mc_uncertainty": float(mc_uncertainty),
            "tta_uncertainty": float(tta_uncertainty),
            "uncertainty_fused": float(fused_uncertainty),
            "model_used": disease_key,
            "cropped_image": cropped_image,
            "boxed_image": boxed,
            "heatmap": heatmap_img,
        }
