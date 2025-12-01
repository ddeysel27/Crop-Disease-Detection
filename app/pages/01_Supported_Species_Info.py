import streamlit as st
import base64
import os

from utils.species_mapping import DISEASE_LABELS
from utils.llm_client import run_llm

st.set_page_config(layout="wide")

st.title("Supported Species for Detection & Classification")
st.write("Click any species button to preview a real leaf sample and see AI-generated details.")
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

def get_image_path(key: str) -> str | None:
    """
    Return the path to the sample image for a species if it exists,
    otherwise None.
    """
    # try JPG and PNG just in case
    candidates = [
        os.path.join(IMAGE_FOLDER, f"{key}.JPG"),
        os.path.join(IMAGE_FOLDER, f"{key}.JPEG"),
        os.path.join(IMAGE_FOLDER, f"{key}.jpg"),
        os.path.join(IMAGE_FOLDER, f"{key}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# -----------------------------
# BUTTON GRID
# -----------------------------
cols = st.columns(4)

# persist selection across reruns
if "selected_species" not in st.session_state:
    st.session_state.selected_species = None

for i, (label, key) in enumerate(species_data):
    col = cols[i % 4]
    with col:
        if st.button(label, key=f"btn_{key}"):
            st.session_state.selected_species = key

# -----------------------------
# DISPLAY AREA
# -----------------------------
st.markdown("---")
st.subheader("Species Information")

selected = st.session_state.selected_species

if selected:
    label = next(lbl for lbl, k in species_data if k == selected)
    diseases = DISEASE_LABELS.get(selected, [])

    img_path = get_image_path(selected)

    img_col, text_col = st.columns([1, 2])

    # Show leaf image on the left if available
    with img_col:
        if img_path:
            st.image(img_path, caption=f"{label} – sample leaf")
        else:
            st.info("No sample image found for this species.")

    # LLM description + diseases on the right
    with text_col:
        with st.spinner("Generating AI description…"):
            llm_text = run_llm(
                f"Provide a short, practical description of the crop species '{label}'. "
                f"Then briefly describe the common leaf diseases from datasets for this species: {diseases}."
            )

        st.write(f"### {label}")
        st.write(llm_text)

else:
    st.write("Click a species button above to see a sample leaf image and AI-generated information here.")
