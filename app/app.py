import streamlit as st

# ---- Page Config ----
st.set_page_config(
    page_title="Crop Doctor",
    page_icon="🌱",
    layout="wide",
)


# ---- Main App ----
st.title("Crop Doctor")
st.write(
    """
    Welcome to the Crop Disease Detection system.

    Use the sidebar to:
    - Upload an image for detection  
    - Search for treatment guidance  
    - Browse model performance  
    """
)

st.divider()

st.subheader("Navigation")
st.write("Use the **left sidebar** to move between pages.")

st.info("Start with **Upload & Classify** to test the pipeline.", icon="⚡")
