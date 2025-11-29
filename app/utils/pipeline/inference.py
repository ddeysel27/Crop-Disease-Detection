import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np


from utils.model_utils import clean_label
from utils.species_mapping import (
    SPECIES_LIST,
    SPECIES_TO_MODEL,
    DISEASE_LABELS
)
from utils.load_model.loaders import load_species_model, load_disease_model

from utils.detection.leaf_detector import LeafDetector

from utils.ood.ood_detector import OODDetector


class InferencePipeline:
    def __init__(self, device="cpu"):
        self.device = device

        # species classifier
        self.species_model = load_species_model(device=self.device)

        # disease classifiers
        self.disease_models = {
            key: load_disease_model(path, device=self.device)
            for key, path in SPECIES_TO_MODEL.items()
        }

        # YOLO detector
        self.detector = LeafDetector("models/yolo_plantdoc_detect.pt")

        # OOD detector
        self.ood = OODDetector()

        # transforms
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def tta_transforms(self, img):
        """Return a list of augmented images for TTA."""
        return [
            img,                                      # original
            TF.hflip(img),                            # horizontal flip
            TF.rotate(img, 10),                       # rotate +10°
            TF.rotate(img, -10),                      # rotate -10°
            TF.adjust_brightness(img, 1.2),           # brighter
            TF.adjust_contrast(img, 1.2),             # higher contrast
        ]
    
    def tta_predict(self, model, img):
        """Run the model on several augmented versions and average outputs."""
        model.eval()
        preds = []

        for augmented in self.tta_transforms(img):
            tensor = self.preprocess(augmented)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)

            preds.append(probs.cpu().numpy())

        preds = np.array(preds)           # shape: [T, 1, num_classes]
        mean_pred = preds.mean(axis=0)[0] # averaged vector
        return mean_pred
    
    def tta_uncertainty(self, model, img):
        preds = []

        for augmented in self.tta_transforms(img):
            tensor = self.preprocess(augmented)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)

            preds.append(probs.cpu().numpy())

        preds = np.array(preds)[:, 0, :]  # [T, classes]
        uncertainty = preds.std(axis=0).mean()

        return float(uncertainty)

    def preprocess(self, img: Image.Image):
        return self.tfms(img).unsqueeze(0).to(self.device)

    def predict_species(self, img_tensor):
        self.species_model.eval()
        with torch.no_grad():
            logits = self.species_model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return idx.item(), conf.item()

    def predict_disease(self, model_key, img_tensor):
        """model_key = 'Cassava', 'Rice', or 'PlantVillage'"""
        model = self.disease_models.get(model_key)
        if model is None:
            return None, 0.0

        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return idx.item(), conf.item()
    
    def mc_dropout_predict(self, model, img_tensor, passes=15):
        model.train()  # enable dropout

        preds = []
        with torch.no_grad():
            for _ in range(passes):
                logits = model(img_tensor)

                # If model outputs tuple (logits, attn) → use only logits
                if isinstance(logits, tuple):
                    logits = logits[0]

                probs = torch.softmax(logits, dim=1)
                preds.append(probs)

        preds = torch.stack(preds)           # [passes, 1, num_classes]
        mean_pred = preds.mean(dim=0)        # [1, num_classes]

        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum().item()

        return mean_pred, entropy


    # Main preeict funtion that runs when in 02_Upload_and_Classify
    def predict(self, image: Image.Image) -> dict:

        # ---------------------------------------------------
        # 0. Leaf Detection (YOLO)
        # ---------------------------------------------------
        cropped_image, boxed = self.detector.detect_leaf(image)
        if cropped_image is None:
            return {"error": "No leaf detected. Please upload a clear photo of a plant leaf."}

        # ---------------------------------------------------
        # 1. Preprocess
        # ---------------------------------------------------
        img_tensor = self.preprocess(cropped_image)

        # ---------------------------------------------------
        # 2. OOD CHECK (species model)
        # ---------------------------------------------------
        with torch.no_grad():
            species_logits = self.species_model(img_tensor)

        if self.ood.is_ood(species_logits)["is_ood"]:
            return {"error": "❗ The image does not look like a valid plant leaf or is out of distribution."}

        # ---------------------------------------------------
        # 3. SPECIES PREDICTION
        # ---------------------------------------------------
        probs = torch.softmax(species_logits, dim=1)
        species_conf, species_idx = probs.max(dim=1)
        species_name = SPECIES_LIST[species_idx]

        # ---------------------------------------------------
        # 4. SELECT DISEASE MODEL
        # ---------------------------------------------------
        if species_name in ["Cassava", "Rice"]:
            disease_key = species_name
        else:
            disease_key = "PlantVillage"

        model = self.disease_models[disease_key]

        # ---------------------------------------------------
        # 5. MC-DROPOUT
        # ---------------------------------------------------
        mc_mean_pred, mc_uncertainty = self.mc_dropout_predict(model, img_tensor)

        # mc_mean_pred: shape (1, C)
        mc_mean_pred = mc_mean_pred.squeeze()
        if mc_mean_pred.ndim != 1:
            mc_mean_pred = mc_mean_pred.view(-1)

        disease_idx = int(mc_mean_pred.argmax())
        disease_conf_mc = float(mc_mean_pred[disease_idx])

        # ---------------------------------------------------
        # 6. TTA
        # ---------------------------------------------------
        tta_mean_pred = self.tta_predict(model, cropped_image)
        tta_uncertainty = self.tta_uncertainty(model, cropped_image)

        # tta_mean_pred: shape (C,)
        if tta_mean_pred.ndim != 1:
            tta_mean_pred = tta_mean_pred.view(-1)

        disease_conf_tta = float(tta_mean_pred[disease_idx])

        # ---------------------------------------------------
        # 7. FUSE MC + TTA
        # ---------------------------------------------------
        fused_conf = (disease_conf_mc + disease_conf_tta) / 2
        fused_uncertainty = (mc_uncertainty + tta_uncertainty) / 2

        # ---------------------------------------------------
        # 8. FINAL LABEL
        # ---------------------------------------------------
        raw_label = DISEASE_LABELS[disease_key][disease_idx]
        disease_label = clean_label(raw_label)

        # ---------------------------------------------------
        # 9. SAVE RESULT FOR LLM
        # ---------------------------------------------------
        try:
            with open("latest_result.txt", "w") as f:
                f.write(f"Species: {species_name}\n")
                f.write(f"Disease: {disease_label}\n")
                f.write(f"Species_Confidence: {species_conf:.4f}\n")
                f.write(f"Disease_Confidence_Fused: {fused_conf:.4f}\n")
                f.write(f"Uncertainty_Fused: {fused_uncertainty:.4f}\n")
        except:
            pass

        # ---------------------------------------------------
        # 10. RETURN CLEAN OUTPUT
        # ---------------------------------------------------
        return {
            "species": species_name,
            "species_conf": float(species_conf),
            "disease": disease_label,
            "disease_conf": fused_conf,
            "mc_uncertainty": float(mc_uncertainty),
            "tta_uncertainty": float(tta_uncertainty),
            "uncertainty_fused": float(fused_uncertainty),
            "model_used": disease_key,
            "cropped_image": cropped_image,
            "boxed_image": boxed
        }
