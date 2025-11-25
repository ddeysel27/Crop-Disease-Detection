import os

# -----------------------------
# Species (from Species_split)
# -----------------------------
SPECIES_SPLIT_PATH = "data/processed/Species_split/train"
SPECIES_LIST = sorted(os.listdir(SPECIES_SPLIT_PATH))

# -----------------------------
# Disease class folders
# -----------------------------
CASSAVA_PATH = "data/processed/Cassava/train"
RICE_PATH = "data/processed/Rice/train"
PLANTVILLAGE_PATH = "data/processed/PlantVillage/train"

CASSAVA_LABELS = sorted(os.listdir(CASSAVA_PATH))
RICE_LABELS = sorted(os.listdir(RICE_PATH))
PLANTVILLAGE_LABELS = sorted(os.listdir(PLANTVILLAGE_PATH))

# -----------------------------
# Model mapping
# -----------------------------
SPECIES_TO_MODEL = {
    "Cassava": "models/cassava_best.pth",
    "Rice": "models/rice_leaf_best.pth",
    "PlantVillage": "models/plant_village_best.pth",
}

# -----------------------------
# Disease label mapping
# -----------------------------
DISEASE_LABELS = {
    "Cassava": CASSAVA_LABELS,
    "Rice": RICE_LABELS,
    "PlantVillage": PLANTVILLAGE_LABELS,
}

# -----------------------------
# Number of classes PER MODEL
# -----------------------------
SPECIES_NUM_CLASSES = {
    "Cassava": len(CASSAVA_LABELS),
    "Rice": len(RICE_LABELS),
    "PlantVillage": len(PLANTVILLAGE_LABELS),
}
