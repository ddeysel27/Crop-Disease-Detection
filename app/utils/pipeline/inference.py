import torch
from PIL import Image
import torchvision.transforms as transforms


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

    def predict(self, image: Image.Image) -> dict:
        
        # 0. Leaf Detection with YOLO
        cropped_image, boxed = self.detector.detect_leaf(image)

        if cropped_image is None:
            return {
                "error": "No leaf detected. Please upload a clear photo of a plant leaf."
            }


        # 1. Preprocess
        img_tensor = self.preprocess(cropped_image)

        # ---------- OOD CHECK ----------
        with torch.no_grad():
            logits = self.species_model(img_tensor)

        ood_info = self.ood.is_ood(logits)

        if ood_info["is_ood"]:
            return {
                "error": "❗ The image does not look like a valid plant leaf or is out of distribution."
            }
        # --------------------------------

        # Species prediction
        probs = torch.softmax(logits, dim=1)
        species_conf, species_idx = probs.max(dim=1)
        species_name = SPECIES_LIST[species_idx]


        logits = self.species_model(img_tensor)
        ood = self.ood.is_ood(logits)

        if ood["is_ood"]:
            return {"error": "The image appears out-of-distribution (not a valid leaf)."}

    
        # 3. Decide which disease model
        if species_name in ["Cassava", "Rice"]:
            disease_model_key = species_name
        else:
            disease_model_key = "PlantVillage"

        # 4. Disease prediction
        disease_idx, disease_conf = self.predict_disease(disease_model_key, img_tensor)

        if disease_idx is None:
            disease_label = "Unknown"
        else:
            raw_label = DISEASE_LABELS[disease_model_key][disease_idx]
            disease_label = clean_label(raw_label)


        # ---------------------------------------------------
        # 5. SAVE OUTPUT FOR LLM
        # ---------------------------------------------------
        try:
            with open("latest_result.txt", "w") as f:
                f.write(f"Species: {species_name}\n")
                f.write(f"Disease: {disease_label}\n")
                f.write(f"Species_Confidence: {species_conf:.2f}\n")
                f.write(f"Disease_Confidence: {disease_conf:.2f}\n")
        except Exception as e:
            print("Failed to save latest_result.txt:", e)

        # ---------------------------------------------------
        # 6. RETURN CLEAN OUTPUT
        # ---------------------------------------------------
        return {
            "species": species_name,
            "species_conf": float(species_conf),
            "disease": disease_label,
            "disease_conf": float(disease_conf),
            "model_used": disease_model_key,
            "cropped_image": cropped_image,
            "boxed_image": boxed
        }
