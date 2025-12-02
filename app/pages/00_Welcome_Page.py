import streamlit as st

st.set_page_config(page_title="Welcome", layout="wide")

st.title("Smarter Agriculture Through Vision & AI")


I_PATH = "C:/Users/User/Desktop/Data Science/Projects/crop-disease-detection/data/welcome_page.jpg"  
st.image(
    I_PATH,
    width=900
)

# ---------- INTRO TEXT ----------
st.write("""
Welcome to **Crop Doctor**, your intelligent companion for diagnosing plant diseases
directly from a photo. Whether you're a farmer, researcher,
""")
         
st.write("""
gardener, or student,
this tool helps you quickly identify **species**, detect **diseases**, and understand
your plant’s health in seconds.
""")

st.subheader("What This Tool Can Do")
st.write("""
- **Upload a leaf image** and get instant analysis  
- **Identify plant species** using deep learning (ViT)  
- **Detect diseases** across Cassava, Rice, and many PlantVillage crops  
- **Assess plant health** with clear status and confidence levels  
- **Warn on low-confidence predictions** and suggest retaking the photo  
""")

st.subheader("How It Works")
st.write("""
1. Upload a clear photo of a single leaf.  
2. The AI predicts the **species** and then the **disease** using specialized models.  
3. You get a readable report with species, disease, confidence, and plant status.  
4. If the image quality is poor, we’ll tell you and recommend a better shot.
""")

st.info("""
 *Best results tips*  
- Use bright, natural lighting  
- Avoid heavy shadows and blur  
- Let the leaf fill most of the frame  
- Use a plain background if possible  
""")

st.markdown("---")
st.subheader("Ready to Begin?")
st.write("Go to **Upload and Classify** in the sidebar to start diagnosing your plants.")
