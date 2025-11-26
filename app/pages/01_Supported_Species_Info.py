import streamlit as st

st.subheader("Supported Species for Detection & Classification")

st.write("""
Our species classifier is trained on a broad set of crops from the **Species_split** dataset.  
The following species can be detected by the pipeline:

### 🌱 **Supported Species (Species Classifier)**
- Apple  
- Blueberry  
- Cassava  
- Cherry (including sour cherry)  
- Corn (Maize)  
- Grape  
- Orange  
- Peach  
- Pepper (Bell)  
- Potato  
- Raspberry  
- Rice  
- Soybean  
- Squash  
- Strawberry  
- Tomato  

---

### 🧪 **Disease Classification Availability**
Although the system can detect all species above, **disease classification will route to the following models accordingly:**:

- **Cassava** → `cassava_best.pth`  
- **Rice** → `rice_leaf_best.pth`  
- **Apple, Corn, Tomato, & The Rest** → `plant_village_best.pth`  

*Other species will return: `"Disease model not available for this species"`.*

---

### 🔄 How the Pipeline Works  
This application uses a **three-stage intelligent detection pipeline**:

1. **🟦 Stage 1 — Leaf Detection (YOLOv8-Seg/Det)**  
   - Locates the leaf in the image  
   - Draws bounding boxes and masks  
   - Crops the leaf region automatically  
   - Improves accuracy by removing background noise (hands, soil, sky, clutter)

2. **🟩 Stage 2 — Species Classification (Vision Transformer)**  
   - The cropped leaf is passed into a ViT model  
   - Predicts: *Cassava / Rice / Apple / Corn / Tomato*

3. **🟥 Stage 3 — Disease Classification (Species-Specific Model)**  
   - The pipeline dynamically selects the correct disease model  
   - Ensures maximum accuracy and prevents cross-species misclassification

---

### 📷 Image Guidelines for Best Results
To get the most accurate prediction:
- Upload a **clear, single-leaf** photo  
- Ensure the leaf is **centered** and not heavily occluded  
- Avoid:
  - Blurry images  
  - Strong shadows  
  - Entire plants / fields  
  - Overexposed lighting  
- Natural daylight works best  

---

### ⚠️ Current Limitations
- Only one leaf per image  
- Only species listed above  
- Severe zoom-out photos (whole tree, branches, soil background) reduce accuracy  
- Multiple leaves together may confuse the detection model  

More species and diseases will be added in future releases.
""")
