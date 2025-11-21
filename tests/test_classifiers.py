# ============================================================
# test_classifiers.py
#
# Full inference pipeline test for:
#  - YOLO leaf detection
#  - Species ViT classifier
#  - Cassava / Rice / PV disease classifiers
#
# Works with your exact folder layout.
# ============================================================
import torch
import torchvision
print(torch.__version__, torchvision.__version__)



import os
import torch
import cv2
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. RESOLVE PROJECT ROOT
#    This file lives in: <root>/tests/
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_DIR = os.path.join(ROOT_DIR, "models")
TEST_DIR  = os.path.join(ROOT_DIR, "tests", "test_images")

print("ROOT_DIR:", ROOT_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("TEST_DIR:", TEST_DIR)


# ------------------------------------------------------------
# 2. MODEL PATHS
# ------------------------------------------------------------
PATH_SPECIES = os.path.join(MODEL_DIR, "species_classifier_vit.pth")
PATH_CASSAVA = os.path.join(MODEL_DIR, "cassava_best.pth")
PATH_RICE    = os.path.join(MODEL_DIR, "rice_leaf_best.pth")
PATH_PV      = os.path.join(MODEL_DIR, "plant_village_best.pth")
PATH_YOLO    = os.path.join(MODEL_DIR, "yolo_plantdoc_detect.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ------------------------------------------------------------
# 3. LOAD YOLO MODEL (DETECT)
# ------------------------------------------------------------
yolo = YOLO(PATH_YOLO)


# ------------------------------------------------------------
# 4. TRANSFORMS FOR ViT MODELS
# ------------------------------------------------------------
IMG_SIZE = 224
vit_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ------------------------------------------------------------
# 5. LOADER FOR ANY ViT MODEL
# ------------------------------------------------------------
def load_vit(path, num_classes):
    model = models.vit_b_16(weights=None)
    model.heads = nn.Linear(model.heads.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


species_model = load_vit(PATH_SPECIES, num_classes=3)
cassava_model = load_vit(PATH_CASSAVA, num_classes=3)
rice_model = load_vit(PATH_RICE, num_classes=3)
pv_model = load_vit(PATH_PV, num_classes=2)

print("Models loaded.")


# ------------------------------------------------------------
# 6. HELPERS: Assist pipeline
# ------------------------------------------------------------
def predict_vit(pil_img, model, labels):
    tensor = vit_tfms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(1).item()
    return labels[pred]


def crop_leaf_from_yolo(image_path):
    """Detect leaf and crop with YOLO"""
    results = yolo(image_path)[0]

    if len(results.boxes) == 0:
        print("⚠️ No leaf detected:", os.path.basename(image_path))
        return None

    box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    img = cv2.imread(image_path)
    crop = img[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    return Image.fromarray(crop)


# ------------------------------------------------------------
# 7. CLASS LABELS (ADJUST IF NEEDED)
# ------------------------------------------------------------
species_labels = ["cassava", "rice", "other"]

cassava_labels = ["healthy", "CMD", "CGM"]
rice_labels    = ["healthy", "blast", "brownspot"]
pv_labels      = ["healthy", "disease"]


# ------------------------------------------------------------
# 8. RUN PIPELINE ON TEST IMAGES
# ------------------------------------------------------------
def run_tests():
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        print("❌ No images found in test_images/.")
        return

    results = []
    print("\n===============================")
    print(" RUNNING MODEL PIPELINE TESTS")
    print("===============================\n")

    for file in files:
        path = os.path.join(TEST_DIR, file)
        print(f"\n🟦 Testing: {file}")

        # Step 1 — detect leaf
        leaf = crop_leaf_from_yolo(path)
        if leaf is None:
            continue

        # Step 2 — species classifier
        species = predict_vit(leaf, species_model, species_labels)
        print("  🔍 Species:", species)

        # Step 3 — route to correct disease model
        if species == "cassava":
            disease = predict_vit(leaf, cassava_model, cassava_labels)
        elif species == "rice":
            disease = predict_vit(leaf, rice_model, rice_labels)
        else:
            disease = predict_vit(leaf, pv_model, pv_labels)

        print("  🦠 Disease:", disease)
        results.append((file, species, disease))

    print("\n===============================")
    print(" FINAL RESULTS")
    print("===============================\n")

    for f, s, d in results:
        print(f"{f:30s} | species: {s:10s} | disease: {d}")

    return results


if __name__ == "__main__":
    run_tests()
