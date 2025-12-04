# 🌱 Crop Disease Detection — End-to-End AI System

An advanced multi-model AI pipeline combining **Vision Transformers (ViT)** for plant species + disease classification, **LLM-powered agronomy assistance**, and a **RAG knowledge retrieval system**, deployed in a clean and interactive **Streamlit** application.

Users can upload a leaf image and instantly receive:

- ✔ **Predicted plant species**
- ✔ **Predicted disease class** (auto-routed to the correct model)
- ✔ **Confidence scores**
- ✔ **LLM-generated biological explanation + treatment guidance**
- ✔ **A RAG chatbot grounded in scientific literature**

---
## 🎥 Demo Video
[![Demo Video](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=https://youtu.be/6DOHdqP1464)

---

# 📌 Features

## 🔍 1. Vision Transformer–Based Classification
- ViT-Base species classifier  
- Cassava, RiceLeaf, and PlantVillage disease classifiers  
- Transfer learning + fine-tuning  
- Consistent preprocessing pipeline  

---

## 🧠 2. Three-Stage ML Architecture
**Stage 1:** Leaf Detection
**Stage 2:** Predict species  
**Stage 3:** Load correct disease model and predict disease

Modular, production-ready design.

---

## 📚 3. RAG-Powered Agriculture Chatbot
- FAISS vector search  
- MiniLM embeddings  
- GPT-based grounded reasoning  
- llama local response (install ollama separately)

---

## 🛠 4. Streamlit Web Application
- Welcome page with hover previews  
- Upload-and-classify workflow  
- Chatbot assistant    
- Clean, modern UI  

---

## 📊 5. Evaluation Tools
- Accuracy
- TTA  
- Heatmap
- Cross-dataset generalization  

---

# 🔬 Advanced ML Concepts (One Slide Version)

### These are the major concepts, already optimized for presentation decks:

- Vision Transformers (ViT) for species + disease detection

- Two-stage hierarchical classification (species → disease)

- Retrieval-Augmented Generation (RAG) with FAISS + MiniLM

- LLM integration for explanations and agronomy support

- Inference optimization (lazy loading, GPU/CPU detection, transform standardization)

- Dataset engineering across multi-source agricultural datasets

# Modular MLOps-ready architecture

# 🛠 Installation
1. Clone the Repository
git clone https://github.com/<ddeysel27>/crop-disease-detection.git
cd crop-disease-detection

2. Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Set Your OpenAI API Key
Option A — Create a .env file:
OPENAI_API_KEY=your_key_here

Option B — Or export it:
export OPENAI_API_KEY="your_key_here"

# ▶️ Run the Streamlit App

#### From the project root:

streamlit run app/app.py


#### The app will open with:

1. Welcome Page
2. Upload & Classify
3. Chatbot
4. Browse Articles

#  How to Use the Pipeline
### Scout the differnt species
When you click a specie button you will see:

- A sample image of the species leaf
- A gpt description of the species selected and diseases on database

### Upload an image

You will receive:

- Predicted species
- Predicted disease
- Confidence scores
- Heatmap

#### Ask the chatbot

The RAG pipeline retrieves scientific text → LLM answers reliably.

# Model Training

Training notebooks are located in:
/notebooks/

### Each notebook includes:

- Dataset loading
- Augmentation + preprocessing
- ViT fine-tuning
- Evaluation
- Export of .pth model

### To start training:

jupyter notebook
Then open any training file.

# Model Evaluation 

Use:

tests/evaluate_model.ipynb on full tests from each dataset


# This notebook provides:

- Accuracy
- Confusion matrices

# 🤝 Contributing

- Fork the repo
- Create a new branch
- Make changes
- Submit a PR

All contributions are welcome.

# 📜 License

MIT License 
