import streamlit as st
from PIL import Image
from utils.pipeline.inference import InferencePipeline

# ------------------------------------------------------
#    PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Upload & Classify", layout="wide")
st.sidebar.title("📤 Upload & Classify")
st.sidebar.write("Upload a leaf image to detect species and disease.")

st.title("🌿 Upload & Classify")
st.write("Upload a plant leaf image and let the AI identify the species and any possible diseases.")

# ------------------------------------------------------
#    LOAD PIPELINE
# ------------------------------------------------------
def load_pipeline():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = InferencePipeline(device="cpu")
    return st.session_state.pipeline

pipeline = load_pipeline()

# ------------------------------------------------------
#    FILE UPLOAD
# ------------------------------------------------------
uploaded = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Image preview card
    with st.container():
        st.image(img, caption="Uploaded Image", use_column_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    #     CLASSIFICATION SECTION
    # ------------------------------------------------------
    st.subheader("🔍 Running Classification...")

    with st.spinner("Analyzing image..."):
        result = pipeline.predict(img)

    st.success("Prediction Complete")

    # ------------------------------------------------------
    #     SPECIES CARD
    # ------------------------------------------------------
    with st.container():
        st.markdown("### 🌱 Species Prediction")

        sp_col1, sp_col2 = st.columns([3, 1])
        with sp_col1:
            st.write(f"**Predicted Species:** {result['species']}")
        with sp_col2:
            st.write(f"**Conf:** {result['species_conf']:.2f}")

        st.progress(result["species_conf"])

        # If low confidence:
        if result["species_conf"] < 0.70:
            st.warning("🟡 Low confidence — try retaking the photo with better lighting and a clear leaf.")

    # ------------------------------------------------------
    #     DISEASE CARD
    # ------------------------------------------------------
    with st.container():
        st.markdown("### 🍂 Disease Prediction")

        dis_col1, dis_col2 = st.columns([3, 1])
        with dis_col1:
            st.write(f"**Disease:** {result['disease']}")
        with dis_col2:
            st.write(f"**Conf:** {result['disease_conf']:.2f}")

        st.write(f"**Model Used:** `{result['model_used']}`")
        st.progress(result["disease_conf"])

        if result["disease_conf"] < 0.70:
            st.warning("🟡 Disease confidence is low — ensure the leaf is centered and the image is sharp.")

    # ------------------------------------------------------
    #     STATUS CARD
    # ------------------------------------------------------
    with st.container():
        st.markdown("### 🩺 Plant Status")

        if "healthy" in result["disease"].lower():
            st.success("🌱 Status: **Healthy** — No major disease detected.")
        else:
            st.error(f"🍂 Status: **Diseased** — {result['disease']}")

    # ------------------------------------------------------
    #     RETAKE PHOTO TIPS
    # ------------------------------------------------------
    st.markdown("---")
    with st.expander("📸 Tips for Better Predictions"):
        st.write("""
        - Ensure the **leaf fills most of the frame**
        - Take photo in **bright natural lighting**
        - Avoid shadows and blurry images  
        - Place leaf on a **plain, solid background**
        - Hold camera **steady and close**
        """)

