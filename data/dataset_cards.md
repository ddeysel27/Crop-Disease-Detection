# 🌱 Dataset Cards — Crop Disease Detection

This document summarizes all datasets currently used for training, validation, and testing in the **Crop Disease Detection — From Lab to Field** project.

Each dataset card includes:  
- 📦 Source & License  
- 🌾 Number of Classes  
- 📊 Approx. Samples (Train/Val/Test)  
- ⚙️ Preprocessing & Notes  

---

## 🧩 1. PlantVillage

**Type:** Controlled lab dataset  
**Source:** [Kaggle – PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)  
**License:** CC BY-SA 4.0  

| Attribute | Details |
|:--|:--|
| **Classes** | 38 (crop–disease combinations) |
| **Images** | ≈ 54,300 RGB images |
| **Train/Val/Test Split** | 70 % / 15 % / 15 % |
| **Image Size** | 256×256 px (resized to 224×224) |
| **Preprocessing** | Center crop → resize → normalize (ImageNet mean/std) |
| **Notes** | Highly controlled lighting and background; used primarily for baseline model pretraining. |

---

## 🍃 2. PlantDoc

**Type:** Real-world field images  
**Source:** [GitHub – PlantDoc Object Detection Dataset](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)  
**License:** CC BY 4.0  

| Attribute | Details |
|:--|:--|
| **Classes** | 38 (same label taxonomy as PlantVillage; field conditions) |
| **Images** | ≈ 2,500 labeled images |
| **Train/Val/Test Split** | 70 % / 15 % / 15 % |
| **Image Size** | Variable (resized to 224×224) |
| **Preprocessing** | Leaf segmentation (SAM 2 / U²-Net), random crop, color jitter, horizontal flip |
| **Notes** | Serves as *domain adaptation* data — brings realism via variable lighting, occlusions, and natural backgrounds. |

---

## 🌿 3. Cassava Leaf Disease Dataset

**Type:** Field dataset  
**Source:** [TensorFlow Datasets – Cassava Leaf Disease](https://www.tensorflow.org/datasets/catalog/cassava)  
**License:** CC BY 4.0  

| Attribute | Details |
|:--|:--|
| **Classes** | 5 (CMD, CBSD, BLS, CGM, Healthy) |
| **Images** | ≈ 21,400 images |
| **Train/Val/Test Split** | 80 % / 10 % / 10 % |
| **Image Size** | 600×600 px (resized to 224×224) |
| **Preprocessing** | Random rotation, brightness/contrast shift, leaf mask extraction |
| **Notes** | High-resolution field imagery; complements PlantDoc for tropical crops (cassava, maize). |

---

## 🌾 4. Rice Leaf Disease Dataset

**Type:** Field dataset  
**Source:** [Mendeley Data – Rice Leaf Disease](https://data.mendeley.com/datasets/tywbtsjrjv/1)  
**License:** CC BY 4.0  

| Attribute | Details |
|:--|:--|
| **Classes** | 4 (Bacterial leaf blight, Brown spot, Leaf smut, Healthy) |
| **Images** | ≈ 1,200 images |
| **Train/Val/Test Split** | 70 % / 15 % / 15 % |
| **Image Size** | 300×300 px (resized to 224×224) |
| **Preprocessing** | Resize → normalization → background removal (U²-Net) |
| **Notes** | Smaller dataset; useful for fine-tuning model generalization to cereal crops. |

---

## 🧠 Integration Notes

- All datasets are normalized to **ImageNet statistics** (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).  
- Augmentation (Albumentations) includes random crop, flip, rotation, blur, brightness, and hue shift.  
- Final dataset balance maintained via oversampling minority disease classes.  
- Metadata stored in `data/processed/metadata.csv` for reproducibility.

---

## 📚 Citation

If you use these datasets, please cite their original authors as indicated on each source page, and acknowledge:

> Deysel, D. (2025). *Crop Disease Detection — From Lab to Field.* University of Miami.

---

Installation Guides:
# 📂 Data Directory

This project uses three datasets:

| Dataset | Source | Download |
|:--|:--|:--|
| **PlantVillage** | Kaggle | [link](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| **PlantDoc** | GitHub / Roboflow | [link](https://public.roboflow.com/object-detection/plantdoc) |
| **Cassava Leaf Disease** | TensorFlow Datasets | [link](https://www.tensorflow.org/datasets/catalog/cassava) |

After downloading, place them under:

data/
├── raw/
│ ├── plantvillage/
│ ├── plantdoc/
│ └── cassava/


---

### 3️⃣ Track data in `.gitignore`
Add this to `.gitignore` (if not already):

Data folders

data/raw/
data/interim/
data/processed/
*.zip
*.tar.gz


---

### 4️⃣ (Optional) Use **DVC** or **Git LFS**
If you *must* version datasets for reproducibility:
- Use **[DVC](https://dvc.org/)** (Data Version Control) — recommended for ML workflows.
- Or use **Git LFS**, but only for small subsets or sample images.

---

### 5️⃣ Push only:
✅ Code (`src/`, `notebooks/`, `app/`)  
✅ Metadata (`dataset_cards.md`, `README.md`)  
✅ Scripts to download/process data  

---

### 🧩 TL;DR
| Push | Don’t Push |
|:--|:--|
| Code, notebooks, configs | Full datasets |
| `requirements.txt`, `dataset_cards.md` | `.jpg`, `.png`, `.tif` files |
| Small sample images (optional) | Zipped datasets or large outputs |

---

If you’d like, I can generate a **clean `.gitignore` and `data/README.md`** template for your repo (formatted for your Crop Disease Detection project) so your teammates can set up the same folder structure safely. Would you like me to make that next?
