import streamlit as st

st.set_page_config(page_title="Crop Doctor", layout="wide")

# ==========================================================
# HEADER BANNER
# ==========================================================
st.markdown(
"""
<div style="background-color:#0E1A2B; padding:60px; border-radius:15px; text-align:center; margin-bottom:40px;">
<h1 style="color:white; font-size:52px; margin-bottom:10px;">The Crop Doctor</h1>
<p style="color:#B3C7E0; font-size:20px; margin-top:0;">
AI-powered crop disease detection and real-time treatment guidance
</p>
</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# CSS
# ==========================================================
st.markdown(
"""
<style>
.card-container {
    display:flex;
    justify-content:space-between;
    gap:25px;
    margin-top:25px;
}
.card {
    background-color:#0E2A47;
    border-radius:16px;
    padding:30px;
    width:23%;
    text-align:center;
    color:white;
    text-decoration:none;
    transition:0.2s ease-in-out;
}
.card:hover {
    transform:translateY(-5px);
    box-shadow:0px 8px 20px rgba(0,0,0,0.4);
}
.card-title {
    font-size:26px;
    font-weight:700;
    margin-bottom:12px;
}
.card-text {
    font-size:16px;
    color:#B3C7E0;
}
</style>
""",
unsafe_allow_html=True
)

# ==========================================================
# CARDS (NO INDENTATION!)
# ==========================================================
st.markdown(
"""
<div class="card-container">

<a href="/Welcome_Page" class="card">
<div class="card-title">🍓 Welcome Page</div>
<div class="card-text">Explore what the application offers.</div>
</a>

<a href="/Upload_and_Classify" class="card">
<div class="card-title">📸 Upload & Classify</div>
<div class="card-text">Upload a leaf image and detect diseases instantly.</div>
</a>

<a href="/Supported_Species_Info" class="card">
<div class="card-title">🌱 Supported Species</div>
<div class="card-text">Browse supported crops & diseases.</div>
</a>

<a href="/Talk_to_Our_Chatbot" class="card">
<div class="card-title">💬 Chatbot Assistant</div>
<div class="card-text">Ask the AI assistant for solutions.</div>
</a>

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# NAVIGATION
# ==========================================================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
"""
<h2 style="color:white;">Navigation</h2>
<p style="color:#B3C7E0; font-size:16px;">Use the left sidebar to move between pages.</p>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div style="background-color:#0E2A47; padding:20px; border-radius:12px;">
<p style="color:#FFC857; font-size:17px; margin:0;">
⚡ Start with <b>Upload & Classify</b> to test the pipeline.
</p>
</div>
""",
unsafe_allow_html=True
)
