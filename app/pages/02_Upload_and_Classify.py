import streamlit as st
from PIL import Image
from utils.pipeline.inference import InferencePipeline
from utils.visuals.uncertainty_badge import render_uncertainty_badge
from utils.visuals.radar_plot import radar_chart

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
    st.session_state.last_uploaded_image = None


st.set_page_config(page_title="Upload & Classify", layout="wide")
st.title("🌿 Upload & Classify")
st.write("Upload a plant leaf image and let the AI identify the species and any possible diseases.")


# ------------------------------------------------------
# LOAD PIPELINE
# ------------------------------------------------------
def load_pipeline():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = InferencePipeline(device="cpu")
    return st.session_state.pipeline

pipeline = load_pipeline()

# ------------------------------------------------------
# FILE UPLOADER
# ------------------------------------------------------
uploaded = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    # Preview image
    st.image(img, caption="Uploaded Image", use_column_width=False)
    st.markdown("---")

    st.subheader("🔍 Running Classification...")

    with st.spinner("Analyzing image..."):
        result = pipeline.predict(img)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    st.success("Prediction Complete")
    st.session_state.last_uploaded_image = img
    st.session_state.last_result = result



    # ------------------------------------------------------
    # LEAF DETECTION (YOLO)
    # ------------------------------------------------------
    st.subheader("Leaf Detection")
    st.image(result["boxed_image"], caption="Detected Leaf Bounding Box", use_column_width=False)

    st.subheader("Cropped Leaf Region")
    st.image(result["cropped_image"], caption="Cropped Leaf for Classification", use_column_width=False)

    st.markdown("---")


    # ------------------------------------------------------
    # SPECIES CARD
    # ------------------------------------------------------
    with st.container():
        st.markdown("### Species Prediction")

        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**Predicted Species:** {result['species']}")
        with c2:
            st.write(f"**Conf:** {result['species_conf']:.2f}")

        st.progress(result["species_conf"])

        if result["species_conf"] < 0.70:
            st.warning("⚠️ Low confidence — try retaking the photo under better lighting.")


    # ------------------------------------------------------
    # DISEASE CARD
    # ------------------------------------------------------
    with st.container():
        st.markdown("### Disease Prediction")

        d1, d2 = st.columns([3, 1])
        with d1:
            st.write(f"**Disease:** {result['disease']}")
        with d2:
            st.write(f"**Conf:** {result['disease_conf']:.2f}")

        st.write(f"**Model Used:** `{result['model_used']}`")
        st.progress(result["disease_conf"])

        # Confidence warnings
        if result["disease_conf"] < 0.70:
            st.warning("⚠️ Low disease confidence — ensure the leaf is centered, clear and bright.")
        
        st.markdown("### Model Explainability (GradCam)")
        st.image(result["heatmap"], caption="What the model focused on")



    st.markdown("---")

    # ------------------------------------------------------
    # SUMMARY CARD (Confidence + Uncertainty)
    # ------------------------------------------------------
    with st.container():
        st.markdown("### Model Confidence Summary")

        colA, colB, colC = st.columns(3)

        with colA:
            st.write("**Fused Confidence**", help="Average of MC-Dropout and TTA confidence.")
            st.write(f"{result['disease_conf']:.4f}")

        with colB:
            st.write("**MC Uncertainty**", help="Variation across dropout-enabled forward passes.")
            st.write(f"{result['mc_uncertainty']:.4f}")

        with colC:
            st.write("**TTA Uncertainty**", help="Variation across test-time augmentations.")
            st.write(f"{result['tta_uncertainty']:.4f}")

        # Render Badge
        render_uncertainty_badge(result["uncertainty_fused"])

    st.markdown("---")

    # ------------------------------------------------------
    # RADAR CHART — Confidence & Uncertainty Overview
    # ------------------------------------------------------
    st.markdown("### Model Reliability Radar Chart")

    radar_fig = radar_chart(
        result["disease_conf"],
        result["mc_uncertainty"],
        result["tta_uncertainty"],
        result["uncertainty_fused"]
    )

    st.pyplot(radar_fig, use_container_width=False)
    st.markdown("---")


    # ------------------------------------------------------
    # STATUS CARD
    # ------------------------------------------------------
    with st.container():
        st.markdown("### Plant Status")

        if "healthy" in result["disease"].lower():
            st.success("🟢 **Healthy — No major disease detected.**")
        else:
            st.error(f"🔴 **Diseased — {result['disease']}**")

    st.markdown("---")


    # ------------------------------------------------------
    # TIPS FOR BETTER INPUT
    # ------------------------------------------------------
    with st.expander("📸 Tips for Better Predictions"):
        st.write("""
        - Ensure the **leaf fills most of the frame**
        - Use **bright natural lighting** for clearer images
        - Avoid shadows and blurry photos  
        - Place the leaf on a **plain background**
        - Hold camera **steady and close**
        """)

