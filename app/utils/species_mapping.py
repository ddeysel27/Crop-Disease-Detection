import os

# ----------------------------------------------------
# Resolve project root (works no matter where imported)
# ----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def p(rel_path: str) -> str:
    """Build absolute paths relative to project root."""
    return os.path.join(PROJECT_ROOT, rel_path)

# ----------------------------------------------------
# DATASET FOLDERS (train splits used to get class names)
# ----------------------------------------------------
CASSAVA_PATH = p("data/processed/Cassava/train")
RICE_PATH = p("data/processed/Rice/train")
PLANTVILLAGE_PATH = p("data/processed/PlantVillage/train")

# Class labels come directly from folder names (correct order)
CASSAVA_LABELS = sorted(os.listdir(CASSAVA_PATH))
RICE_LABELS = sorted(os.listdir(RICE_PATH))
PLANTVILLAGE_LABELS = sorted(os.listdir(PLANTVILLAGE_PATH))

# ----------------------------------------------------
# SPECIES LIST (MUST match species classifier training order)
# ----------------------------------------------------
SPECIES_LIST = [
    "Apple",
    "Blueberry",
    "Cassava",
    "Cherry_(including_sour)",
    "Corn_(maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper_bell",
    "Potato",
    "Raspberry",
    "Rice",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]

# ----------------------------------------------------
# SPECIES → DISEASE MODEL KEY
# (three model keys: 'Cassava', 'Rice', 'PlantVillage')
# ----------------------------------------------------
SPECIES_TO_MODEL_KEY = {
    "Cassava": "Cassava",
    "Rice": "Rice",
}

# All non-Cassava, non-Rice species → PlantVillage model
for sp in SPECIES_LIST:
    if sp not in SPECIES_TO_MODEL_KEY:
        SPECIES_TO_MODEL_KEY[sp] = "PlantVillage"

# ----------------------------------------------------
# DISEASE MODEL FILE PATHS (three models total)
# ----------------------------------------------------
DISEASE_MODEL_PATHS = {
    "Cassava": p("models/cassava_best.pth"),
    "Rice": p("models/rice_leaf_best.pth"),
    "PlantVillage": p("models/plant_village_best.pth"),
}

# ----------------------------------------------------
# MODEL KEY → DISEASE LABELS (class names for each model)
# ----------------------------------------------------
DISEASE_LABELS = {
    "Cassava": CASSAVA_LABELS,          # 5 classes
    "Rice": RICE_LABELS,                # 6 classes (from your screenshot)
    "PlantVillage": PLANTVILLAGE_LABELS,  # ~38 classes
}

# ----------------------------------------------------
# CLASS COUNTS (useful for debugging / sanity checks)
# ----------------------------------------------------
SPECIES_NUM_CLASSES = {
    "Cassava": len(CASSAVA_LABELS),
    "Rice": len(RICE_LABELS),
    "PlantVillage": len(PLANTVILLAGE_LABELS),
}
