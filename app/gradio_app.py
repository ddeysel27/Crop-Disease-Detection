import gradio as gr, torch, json, os
from PIL import Image
from pathlib import Path
from src.models.infer import load_model, predict_topk
from app.prompts import SYSTEM_PROMPT

# Optional RAG imports are gated for environments without FAISS/SBERT
def rag_answer(query_text):
    try:
        import faiss, json
        from sentence_transformers import SentenceTransformer
        import numpy as np
        meta = json.loads(Path("app/kb_meta.json").read_text())
        index = faiss.read_index("app/kb.index")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q = model.encode([query_text], normalize_embeddings=True)
        D,I = index.search(q, 3)
        ctxs = [meta["corpus"][i] for i in I[0]]
        cites = [meta["meta"][i] for i in I[0]]
        joined = "\n\n".join(ctxs)
        refs = "; ".join([f"{c['source']}#chunk{c['chunk']}" for c in cites])
        return joined, refs
    except Exception as e:
        return "", ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "experiments/runs/best_model.pth"
LABELS_PATH = "experiments/runs/label_mapping.json"
ARCH = "vit_base_patch16_224"

def model_ready():
    return Path(MODEL_PATH).exists() and Path(LABELS_PATH).exists()

if model_ready():
    model, idx2label = load_model(MODEL_PATH, LABELS_PATH, arch=ARCH, device=DEVICE)
else:
    model, idx2label = None, None

KNOWLEDGE = json.loads(Path("app/treatment_knowledge.json").read_text())

def advise(crop, disease):
    key = f"{crop}::{disease}".lower()
    if key in KNOWLEDGE:
        return KNOWLEDGE[key]
    return {
      "summary": "Isolate affected plants; remove heavily infected leaves; avoid wetting foliage; improve airflow.",
      "possible_actives": ["copper (bacterial/fungal)", "sulfur (fungal)"],
      "cautions": ["Always check local regulations & product labels.", "Consult local extension services."]
    }

def predict_and_chat(img: Image.Image):
    if model is None:
        return ("Model not found. Train a model and save weights at "
                "experiments/runs/best_model.pth with label_mapping.json."), ""
    probs, labels = predict_topk(model, img, idx2label, k=3, device=DEVICE)
    top = labels[0]
    crop, dis = (top.split('-',1)+['unknown'])[:2] if '-' in top else ("unknown","unknown")
    base = advise(crop, dis)

    # Build query for RAG
    query = f"{crop} {dis} treatment prevention extension bulletin best practices"
    ctx, refs = rag_answer(query)

    md = f"""
**Prediction (Top‑3)**  
1) **{labels[0]}** — {probs[0]:.1%}  
2) {labels[1]} — {probs[1]:.1%}  
3) {labels[2]} — {probs[2]:.1%}

**Guidance (concise):** {base['summary']}
**Possible actives:** {', '.join(base.get('possible_actives', []))}  
**Cautions:** {', '.join(base.get('cautions', []))}

**Retrieved context (for the chatbot to cite):**  
{ctx if ctx else '_No KB loaded yet. Add .txt files to app/kb_texts and run app/build_faiss.py_'}

**References:** {refs if refs else '_N/A_'}

> ⚠️ Educational use only. Confirm with local experts and product labels.
"""
    return md, top

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Crop Disease Assistant — Classifier + RAG Chat")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload a leaf photo")
        out = gr.Markdown(label="Diagnosis & Guidance")
    btn = gr.Button("Analyze")
    hidden = gr.Textbox(visible=False)  # top label for potential UI chaining
    btn.click(fn=predict_and_chat, inputs=img, outputs=[out, hidden])

if __name__ == "__main__":
    demo.launch()
