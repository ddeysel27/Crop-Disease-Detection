import os
import cv2
import shutil
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("Using PROJECT_ROOT:", PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "test_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 5   # select top 5 images per disease


# ============================================================
# QUALITY SCORING
# ============================================================
def score_image_quality(path):
    """Compute a weighted image quality score based on sharpness, brightness, contrast."""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return -1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()
        contrast = gray.std()

        # Weighted score
        return 0.6 * sharpness + 0.2 * brightness + 0.2 * contrast

    except Exception:
        return -1


# ============================================================
# GATHER IMAGES FROM LABEL CSV (each dataset has unique format)
# ============================================================
def gather_images(dataset_name, csv_path):
    df = pd.read_csv(csv_path)
    records = []

    # Cassava & RiceLeaf have split column → filter test
    if "split" in df.columns:
        df = df[df["split"] == "test"]

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # RICELEAF FORMAT (has full filepath)
        # ----------------------------------------------------
        if dataset_name.lower() == "riceleaf" and "filepath" in df.columns:
            img_path = row["filepath"]
            label = row["label"]

        # ----------------------------------------------------
        # CASSAVA FORMAT (image_id only, need to reconstruct path)
        # ----------------------------------------------------
        elif dataset_name.lower() == "cassava" and "image_id" in df.columns:
            image_id = row["image_id"]
            label = row["label_name"]

            # cassava folder path: data/processed/Cassava/test/<label>/<image_id>
            cassava_root = os.path.join(PROJECT_ROOT, "data", "processed", "Cassava", "test")
            img_path = os.path.join(cassava_root, label, image_id)

        # ----------------------------------------------------
        # PLANTVILLAGE FORMAT (full filepath)
        # ----------------------------------------------------
        elif dataset_name.lower() == "plantvillage" and "filepath" in df.columns:
            img_path = row["filepath"]
            label = row["label"]

        else:
            print(f"⚠️ Skipping unsupported row format in {dataset_name}")
            continue

        # ensure exists
        if not os.path.exists(img_path):
            # Skip quietly if missing
            continue

        records.append({
            "dataset": dataset_name,
            "label": label,
            "image": img_path
        })

    return records


# ============================================================
# SELECT BEST IMAGES BASED ON QUALITY
# ============================================================
def select_best_images(records, top_k=TOP_K):
    if not records:
        return []

    df = pd.DataFrame(records)
    df["quality"] = df["image"].apply(score_image_quality)

    selected = []

    for (dataset, label), group in df.groupby(["dataset", "label"]):
        group_sorted = group.sort_values("quality", ascending=False)
        top_rows = group_sorted.head(top_k)

        for _, row in top_rows.iterrows():
            selected.append(row.to_dict())

    return selected


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    DATASETS = {
        "Cassava": os.path.join(PROJECT_ROOT, "data", "processed", "Cassava", "labels.csv"),
        "RiceLeaf": os.path.join(PROJECT_ROOT, "data", "processed", "Rice", "labels.csv"),
        "PlantVillage": os.path.join(PROJECT_ROOT, "data", "processed", "PlantVillage", "labels.csv"),
        # PlantDoc intentionally EXCLUDED (no test labels)
    }

    all_records = []

    # ----------------------
    # Load each dataset CSV
    # ----------------------
    for name, csv_path in DATASETS.items():
        print(f"\nScanning {name}...")

        if not os.path.exists(csv_path):
            print("Missing:", csv_path)
            continue

        recs = gather_images(name, csv_path)
        print(f" Found {len(recs)} labeled images in {name}")
        all_records.extend(recs)

    # ----------------------
    # Select best images
    # ----------------------
    print("\nSelecting best images...")
    best = select_best_images(all_records, TOP_K)
    print(f"  ✔ Selected {len(best)} high-quality images total")

    # ----------------------
    # Save images
    # ----------------------
    print("\nSaving to:", OUTPUT_DIR)

    for rec in best:
        src = rec["image"]
        base = os.path.basename(src)
        dst = os.path.join(
            OUTPUT_DIR,
            f"{rec['dataset']}_{rec['label']}_{base}"
        )
        shutil.copy(src, dst)

    print("\nDONE! Check folder:")
    print("  →", OUTPUT_DIR)
