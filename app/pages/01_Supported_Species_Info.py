import streamlit as st
import base64
import os

from utils.species_mapping import DISEASE_LABELS
from utils.llm_client import run_llm

st.set_page_config(layout="wide")

st.title("🌱 Supported Species for Detection & Classification")
st.write("Hover over any species button to preview a real leaf sample.")
st.write("Click a species to view AI-generated description + diseases we detect.")
st.markdown("---")

# -----------------------------
# Species definitions
# -----------------------------
species_data = [
    ("🍎 Apple", "apple"),
    ("🫐 Blueberry", "blueberry"),
    ("🌿 Cassava", "cassava"),
    ("🍒 Cherry (including sour)", "cherry"),
    ("🌽 Corn (Maize)", "corn"),
    ("🍇 Grape", "grape"),
    ("🍊 Orange", "orange"),
    ("🍑 Peach", "peach"),
    ("🫑 Pepper (Bell)", "pepperbell"),
    ("🥔 Potato", "potato"),
    ("🍓 Raspberry", "raspberry"),
    ("🌾 Rice", "rice"),
    ("🌱 Soybean", "soybean"),
    ("🎃 Squash", "squash"),
    ("🍓 Strawberry", "strawberry"),
    ("🍅 Tomato", "tomato"),
]

IMAGE_FOLDER = "app/species_images"

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


st.markdown("""
<style>
.species-wrapper {
    position: relative;
}

.species-btn {
    width: 100%;
    padding: 8px 16px;
    background: #0b1220;
    border: 1px solid #1f2937;
    border-radius: 12px;
    color: #e5e7eb;
    font-size: 0.95rem;
    text-align: center;
    cursor: pointer;
}

.species-btn:hover {
    border-color: #38bdf8;
    background: #0f172a;
}

/* Hover image */
.preview-image {
    position: absolute;
    top: 115%;
    left: 50%;
    transform: translateX(-50%);
    width: 180px;
    height: 180px;
    border-radius: 12px;
    background-size: cover;
    background-position: center;
    border: 1px solid #1f2937;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease-out;
    box-shadow: 0 18px 40px rgba(0,0,0,0.65);
    z-index: 9999;
}

.species-wrapper:hover .preview-image {
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# GRID BUTTONS USING REAL STREAMLIT BUTTONS
# -----------------------------
cols = st.columns(4)

selected_species = None

for i, (label, key) in enumerate(species_data):
    col = cols[i % 4]

    with col:
      img_path = os.path.join(IMAGE_FOLDER, f"{key}.JPG")
      preview_html = ""

      if os.path.exists(img_path):
         b64 = img_to_base64(img_path)
         preview_html = f"""
               <div class="preview-image"
                  style="background-image:url('data:image/jpeg;base64,{b64}')"></div>
         """

      # HTML version – hover works and click triggers Streamlit
      html = f"""
      <div class="species-wrapper">
         <button class="species-btn" onclick="document.getElementById('species_{key}').click()">
            {label}
         </button>

         <div class="preview-image" 
               style="background-image:url('data:image/jpeg;base64,{b64}')">
         </div>

         <form method="post">
            <input id="species_{key}" type="submit" name="species" value="{key}" style="display:none;">
         </form>
      </div>
      """

      st.markdown(html, unsafe_allow_html=True)


# -----------------------------
# Show species info
# -----------------------------
st.markdown("---")
st.subheader("📘 Species Information")

if selected_species:
    label = next(lbl for lbl, k in species_data if k == selected_species)
    diseases = DISEASE_LABELS.get(selected_species, [])

    with st.spinner("Generating AI description…"):
        llm_text = run_llm(
            f"Provide a short, practical description of the crop species '{label}'. "
            f"Then briefly describe the common leaf diseases from datasets for this species: {diseases}."
        )

    st.write(f"### 🌿 {label}")
    st.write(llm_text)
else:
    st.write("Click a species button above to see AI-generated information here.")
