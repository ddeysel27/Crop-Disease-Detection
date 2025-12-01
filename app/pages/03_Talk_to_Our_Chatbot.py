import streamlit as st
import requests
import json
import os

# =========================================================
#                      PAGE SETUP
# =========================================================
st.set_page_config(page_title="Crop Care Assistant", layout="wide")
st.title("Talk To Our Chatbot")
st.write("Ask questions about your plant diagnosis, treatment steps, or prevention tips.")

# =========================================================
#             LOAD LAST PIPELINE RESULT (TXT FILE)
# =========================================================
def load_last_prediction():
    """Loads the species + disease output saved by inference.py."""
    try:
        if os.path.exists("latest_result.txt"):
            return open("latest_result.txt", "r").read()
        return ""
    except:
        return ""

initial_context = load_last_prediction()

st.info("The assistant automatically uses the latest analysis from the Upload & Classify page.")

# =========================================================
#               OLLAMA LOCAL MODEL CALL
# =========================================================
def ask_ollama(prompt, model="llama3.2"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True
        )

        full_reply = ""

        for line in response.iter_lines():
            if not line:
                continue

            try:
                json_obj = json.loads(line.decode("utf-8"))
            except:
                continue

            full_reply += json_obj.get("response", "")

            if json_obj.get("done"):
                break

        return full_reply.strip()

    except Exception as e:
        return f"⚠️ Error contacting local LLM: {e}"

# =========================================================
#                   CHAT HISTORY STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for speaker, msg in st.session_state.history:
    st.chat_message(speaker).write(msg)

# =========================================================
#             SYSTEM PROMPT (CONTEXT + BEHAVIOR)
# =========================================================
def build_system_prompt(user_msg, diagnosis_text):
    return f"""
You are an agricultural plant-health assistant trained to interpret leaf-disease diagnoses.

You ALWAYS base your answers on the diagnosis information below:

LATEST DIAGNOSIS:
{diagnosis_text}

TASK:
- Explain the disease in simple terms
- Provide treatment steps
- Provide prevention and care instructions
- If the plant is healthy, give general care tips
- If the disease confidence is low, recommend retaking a better photo
- If the user uploads irrelevant questions, politely redirect to plant care

USER QUESTION:
{user_msg}

RESPOND CLEARLY AND PROFESSIONALLY.
"""


# =========================================================
#                      CHAT INPUT
# =========================================================
user_msg = st.chat_input("Ask about the diagnosis, treatment, or plant care...")

if user_msg:
    # Show user message
    st.session_state.history.append(("user", user_msg))
    st.chat_message("user").write(user_msg)

    # Build full prompt
    final_prompt = build_system_prompt(user_msg, initial_context)

    # Call local model
    with st.spinner("Thinking..."):
        ai_reply = ask_ollama(final_prompt)

    # Save assistant message
    st.session_state.history.append(("assistant", ai_reply))
    st.chat_message("assistant").write(ai_reply)
