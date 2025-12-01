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


# -----------------------------
# CSS to enable hover image preview
# -----------------------------
css = """
<style>
.species-button {
    position: relative;
}

.species-button:hover .preview-image {
    opacity: 1;
}

.preview-image {
    position: absolute;
    top: 110%;
    left: 50%;
    transform: translateX(-50%);
    width: 180px;
    height: 180px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    background-size: cover;
    background-position: center;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease-out;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
    z-index: 9999;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -----------------------------
# GRID BUTTONS USING REAL STREAMLIT BUTTONS
# -----------------------------
cols = st.columns(4)

selected_species = None

for i, (label, key) in enumerate(species_data):
    col = cols[i % 4]

    with col:
        img_path = os.path.join(IMAGE_FOLDER, f"{key}.JPG")
        img_html = ""

        if os.path.exists(img_path):
            b64 = img_to_base64(img_path)
            img_html = f"""
                <div class="preview-image" style="background-image:url('data:image/jpeg;base64,{b64}')"></div>
            """

        # wrapper div (allows hover)
        st.markdown(f"<div class='species-button'>", unsafe_allow_html=True)

        # REAL Streamlit button → works 100%
        if st.button(label, key=f"btn_{key}"):
            selected_species = key

        # attach preview image
        st.markdown(img_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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
