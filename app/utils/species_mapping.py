import os

# ----------------------------------------------------
# Resolve project root no matter where file is called
# ----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def p(rel_path):
    """Build absolute paths relative to project root."""
    return os.path.join(PROJECT_ROOT, rel_path)

# ----------------------------------------------------
# DATASET FOLDERS
# ----------------------------------------------------
CASSAVA_PATH = p("data/processed/Cassava/train")
RICE_PATH = p("data/processed/Rice/train")
PLANTVILLAGE_PATH = p("data/processed/PlantVillage/train")

# ----------------------------------------------------
# LOAD DISEASE LABEL FOLDERS
# ----------------------------------------------------
CASSAVA_LABELS = sorted(os.listdir(CASSAVA_PATH))
RICE_LABELS = sorted(os.listdir(RICE_PATH))
PLANTVILLAGE_LABELS = sorted(os.listdir(PLANTVILLAGE_PATH))

# ----------------------------------------------------
# SPECIES LIST
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
    "Pepper,_bell",
    "Potato",
    "Raspberry",
    "Rice",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]


# ----------------------------------------------------
# MAP SPECIES → MODEL WEIGHTS
# ----------------------------------------------------
SPECIES_TO_MODEL = {
    "Cassava": p("models/cassava_best.pth"),
    "Rice": p("models/rice_leaf_best.pth"),
    "PlantVillage": p("models/plant_village_best.pth"),
}

# ----------------------------------------------------
# MAP SPECIES → DISEASE LABELS
# ----------------------------------------------------
DISEASE_LABELS = {
    "Apple": ["Apple Scab", "Black Rot", "Cedar Rust"],
    "Blueberry": ["Healthy"], 
    "Cassava": ["CBB", "CBSD", "CGM", "CMD"],
    "Cherry": ["Healthy"],
    "Corn": ["Common Rust", "Northern Leaf Blight", "Gray Leaf Spot"],
    "Grape": ["Black Rot", "Black Measles", "Leaf Blight"],
    "Orange": ["Citrus Greening"],
    "Peach": ["Bacterial Spot"],
    "Pepperbell": ["Bacterial Spot"],
    "Potato": ["Early Blight", "Late Blight"],
    "Raspberry": ["Healthy"],
    "Rice": ["Brown Spot", "Leaf Blight", "Neck Blast"],
    "Soybean": ["Healthy"],
    "Squash": ["Powdery Mildew"],
    "Strawberry": ["Leaf Scorch"],
    "Tomato": [
        "Bacterial Spot", "Early Blight", "Late Blight",
        "Leaf Mold", "Septoria", "Spider Mites",
        "Target Spot", "Yellow Leaf Curl Virus",
        "Mosaic Virus", "Healthy"
    ],
}


# ----------------------------------------------------
# CLASS COUNTS
# ----------------------------------------------------
SPECIES_NUM_CLASSES = {
    "Cassava": len(CASSAVA_LABELS),
    "Rice": len(RICE_LABELS),
    "PlantVillage": len(PLANTVILLAGE_LABELS),
}
