# Crop Disease Detection — End-to-End AI System

This repository contains a modular, production-ready system for detecting plant species and leaf diseases using:

- Vision Transformers (ViT) for species and disease classification
- A hierarchical three-stage inference pipeline
- A Retrieval-Augmented Generation (RAG) agronomy assistant
- A clean and interactive Streamlit application
- A knowledge-grounded chatbot for treatment recommendations and biological insights

Users can upload a leaf image and receive:

- Predicted plant species
- Predicted disease class (automatically routed to the correct model)
- Confidence scores
- Model explanations and treatment guidance via an LLM
- Scientifically grounded RAG responses

---

## 1. Project Overview

This project integrates computer vision, RAG-based NLP, and interactive UI design to create a practical tool for agricultural diagnostics.

The system includes:

- Vision Transformer classification models
- Three-stage hierarchical inference
- RAG pipeline with FAISS and MiniLM
- LLM reasoning for agronomy support
- Streamlit deployment
- Reproducible model training and evaluation workflows

The architecture is designed to be modular, interpretable, and scalable.

---

## 2. Pipeline Architecture
```bash
Leaf Image ──► Preprocessing ──► Species Classifier (ViT)
│
▼
Leaf Detection Model (YOLO) Returns bounding boxes and cropped area
│
▼
Species Classifier (ViT)│   Routes to correct Disease Classifier
│
▼
Disease Classifier (ViT)    Returns disease
│
▼
Results + Confidence Scores
│
├────────► LLM Explanation
│
└────────► RAG Chatbot Response
```

## 2.2 Streamlit Architecture
```bash
App                 (Main application on startup)
│
▼
Welcome Page        (Informative and provides proper instructions for
                    improving photo quality to maximize model accuracy)
│
▼
Species Info Page   (Displays sample leaf images of the corresponding 
                    species button and generates a GPT description of 
                    the selected species. Providing user as much context
                    about species as needed)
│
▼
Upload & Classify   (Page where above pipeline is implemented. Select 
                    photo and get instant results.)
│
▼
Chatbot             (Provides helpful information about possible treatment
                    options, diseases, climate preferences etc...)
```


---

## 3. Vision Transformer Classification

### 3.1 Species Classification
- ViT-Base model trained on multi-species datasets
- Image normalization and standardized transforms
- High recall for routing to correct disease model

### 3.2 Disease Classification
- Separate ViT-Base models for:
  - Cassava diseases
  - RiceLeaf diseases
  - PlantVillage diseases
- Fine-tuned using transfer learning
- Consistent preprocessing across all models

---

## 4. Three-Stage ML Architecture

Stage 1: Leaf Detection  
Stage 2: Species Classification  
Stage 3: Automatic load of correct disease model and disease prediction  

This modular routing architecture enables scalability to new species and diseases without retraining the entire system.

---

## 5. Retrieval-Augmented Generation (RAG)

- FAISS vector store indexing agricultural research text
- MiniLM embeddings for efficient retrieval
- LLM-generated explanations grounded in retrieved context
- Optional support for local models via Ollama

The RAG component ensures biologically accurate, citation-backed responses.

---

## 6. Streamlit Application

### Features include:

- Welcome page with species previews
- Upload-and-classify workflow
- Real-time prediction with confidence scores
- Heatmap visualization (optional Grad-CAM)
- RAG-backed chatbot for treatment recommendations
- A clean, modular UI

---

## 7. Evaluation and Testing

### Tools include:

- Accuracy metrics
- Test-Time Augmentation (TTA)
- Cross-dataset generalization checks
- Confusion matrices
- Heatmap visualization for explainability

### Full evaluation notebook available in:
```bash
tests/evaluate_model.ipynb
```

---

## 8. Installation

### Clone the repository:
```bash
git clone https://github.com/
<ddeysel27>/crop-disease-detection.git
cd crop-disease-detection
```

### Create a virtual environment:
```bash
python -m venv .venv
```

### Activate environment:
```bash
Windows:   .\.venv\Scripts\activate

Mac/Linux: source .venv/bin/activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Set OpenAI API key:
```bash
Option A (preferred):
Create a .env file containing:
OPENAI_API_KEY=your_key_here
```

### Option B:
```bash
export OPENAI_API_KEY="your_key_here"
```
---

## 9. Running the Application

### From the project root:
```bash
streamlit run app/app.py
```

### The app contains:

1. Welcome Page  
2. Upload and Classify  
3. Chatbot  
4. Scientific Article Browser  

---

## 10. Training

Training notebooks are located in:
```bash
/notebooks/
```

### Each notebook includes:

- Dataset loading
- Augmentation and preprocessing
- ViT fine-tuning
- Model evaluation
- Export of .pth weights

### To begin training:
```bash
jupyter notebook
```  
Open any training notebook.

---

## 11. Model Evaluation

### Use:
```bash
tests/evaluate_model.ipynb
```

### This notebook provides:

- Accuracy
- Confusion matrices
- TTA performance
- Cross-dataset generalization results

---

## 12. Contributing

1. Fork the repository  
2. Create a new branch  
3. Make changes  
4. Submit a pull request  

*Contributions of new species, diseases, or model improvements are encouraged.*

---

## 13. License

This project is released under the MIT License.
