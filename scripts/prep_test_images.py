import os
import random
import shutil

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

OUTPUT_DIR = os.path.join(ROOT_DIR, "tests", "test_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_IMAGES_PER_CLASS = 2   # choose how many per class to copy


# --------------------------------------------------
# DATASET PATHS (YOUR EXACT STRUCTURE)
# --------------------------------------------------
DATASETS = {
    "cassava": os.path.join(ROOT_DIR, "data", "processed", "cassava", "test"),
    "riceleaf": os.path.join(ROOT_DIR, "data", "processed", "riceleaf", "test"),
    "plantVillage": os.path.join(ROOT_DIR, "data", "processed", "plantVillage", "test"),

    # PlantDoc special case: images/ contains jpg/png
    "PlantDoc": os.path.join(ROOT_DIR, "data", "processed", "PlantDoc", "test", "images"),
}

def is_image(f):
    return f.lower().endswith((".jpg", ".jpeg", ".png"))


# --------------------------------------------------
# SCRIPT
# --------------------------------------------------

print("\n Saving sample test images to:", OUTPUT_DIR)

for dataset_name, path in DATASETS.items():

    print(f"\nDataset: {dataset_name}")
    if not os.path.exists(path):
        print(f"  ⚠️ Path not found: {path}")
        continue

    # --------------------------------------------------
    # PlantDoc: images folder → no class subdivision
    # --------------------------------------------------
    if dataset_name == "PlantDoc":
        imgs = [f for f in os.listdir(path) if is_image(f)]
        sample = random.sample(imgs, min(len(imgs), N_IMAGES_PER_CLASS * 3))

        for img in sample:
            src = os.path.join(path, img)
            dst = os.path.join(OUTPUT_DIR, f"{dataset_name}_{img}")
            shutil.copy(src, dst)

        print(f"  ✔ Copied {len(sample)} PlantDoc images.")
        continue

    # --------------------------------------------------
    # Other datasets: class folders → sample per class
    # --------------------------------------------------
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)

        if not os.path.isdir(class_path):
            continue

        imgs = [f for f in os.listdir(class_path) if is_image(f)]
        if len(imgs) == 0:
            continue

        sample = random.sample(imgs, min(N_IMAGES_PER_CLASS, len(imgs)))

        for img in sample:
            src = os.path.join(class_path, img)
            dst_name = f"{dataset_name}_{class_name}_{img}"
            dst = os.path.join(OUTPUT_DIR, dst_name)
            shutil.copy(src, dst)

        print(f" {class_name}: {len(sample)} images copied.")

print("\n DONE! Check folder:")
print("   →", OUTPUT_DIR)
