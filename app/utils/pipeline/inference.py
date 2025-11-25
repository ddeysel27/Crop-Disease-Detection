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


class InferencePipeline:
    def __init__(self, device: str = "cpu"):
        self.device = device

        # Load species classifier
        self.species_model = load_species_model(device=self.device)

        # Load disease classifiers
        self.disease_models = {
            key: load_disease_model(path, device=self.device)
            for key, path in SPECIES_TO_MODEL.items()
        }

        # transforms
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
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
        # ---------------------------
        # 1. Preprocess
        # ---------------------------
        img_tensor = self.preprocess(image)

        # ---------------------------
        # 2. Species prediction
        # ---------------------------
        species_idx, species_conf = self.predict_species(img_tensor)
        species_name = SPECIES_LIST[species_idx]

        # ---------------------------
        # 3. Decide which disease model
        # ---------------------------
        if species_name in ["Cassava", "Rice"]:
            disease_model_key = species_name
        else:
            disease_model_key = "PlantVillage"

        # ---------------------------
        # 4. Disease prediction
        # ---------------------------
        disease_idx, disease_conf = self.predict_disease(disease_model_key, img_tensor)

        if disease_idx is None:
            disease_label = "Unknown"
        else:
            raw_label = DISEASE_LABELS[disease_model_key][disease_idx]
            disease_label = clean_label(raw_label)

        # ---------------------------
        # 5. Return clean output
        # ---------------------------
        return {
            "species": species_name,
            "species_conf": float(species_conf),

            "disease": disease_label,
            "disease_conf": float(disease_conf),

            "model_used": disease_model_key,
        }
