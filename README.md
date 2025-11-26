# Crop Disease Detection — AI for Early Plant Health Diagnostics
From Leaf Images to Actionable Insights — Bridging Machine Learning and Real-World Agriculture
### Overview

Crop Disease Detection is a complete AI system for analyzing plant leaf images, identifying the species, detecting diseases, and presenting actionable information to the user.

The platform combines:

Computer vision models (Vision Transformers + YOLO detector)

Retrieval-augmented knowledge search

Conversational LLMs for treatment advice

A Streamlit application for user-friendly interaction

The goal is to support farmers, agronomists, and agricultural extension officers with fast, trustworthy, and accessible plant-health diagnostics.

### System Architecture
1. Image Processing

YOLO Detector (optional preprocessing)

Used to detect and isolate leaf regions

Helps improve classification on noisy images

Data Augmentation using Albumentations

Lighting variations

Rotations, flips

Blur / contrast adjustments

Increases model robustness

(Segmentation will be added in future versions.)

2. Classification Pipeline

Your system uses two-stage classification:

a. Species Classification

Vision Transformer (ViT-Base)

Trained on your Species_split dataset

Supports >20 species (PlantVillage + Cassava + Rice)

b. Disease Classification

Three specialized models:

Species Group	Model Used
Cassava	Cassava Disease ViT
Rice	Rice Disease ViT
Everything else	PlantVillage Disease ViT

Each model outputs:

Disease label

Confidence score

Calibrated probabilities (temperature scaling ready)

3. Knowledge Retrieval Layer

FAISS vector store for agronomy texts

Sentence-Transformers (MiniLM) for embeddings

Retrieves:

Disease descriptions

Treatment considerations

Cultivation best practices

Preventive measures

4. Conversational LLM Layer

Tiny LLaMA / Mistral (local or HF API)

Converts predictions + retrieved knowledge into:

Simple explanations

Treatment guidance

Step-by-step actions

Warnings/edge cases

The LLM never gives harmful or prescriptive advice; instead it provides neutral, factual, evidence-grounded guidance.

5. User Interface (Streamlit App)

The app includes:

### Upload & Classify

Image preview

Species prediction

Disease prediction

Confidence scoring

Clean label formatting

Plant health assessment

Low-confidence warnings

Retake-photo tips

### Talk to Our Chatbot

Ask for treatment help

Search for similar cases

Retrieve best practices

### Browse Articles

FAISS-powered RAG search

Quickly find relevant agronomy info

### Welcome Page

Explanation of system

Instructions and tips

### Quickstart
1. Create Environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

2. Prepare Data

Place raw datasets inside:

data/raw/


Run preprocessing:

notebooks/01_prepare_splits_preprocess.ipynb


Outputs go to:

data/processed/train
data/processed/val
data/processed/test

3. Train Species & Disease Models
python -m src.models.train \
    --data data/processed \
    --model vit_base_patch16_224 \
    --epochs 15 \
    --img 224

4. Evaluate Models

Open:

notebooks/03_evaluate_report.ipynb


Includes accuracy, F1, confusion matrix, attention maps, etc.

5. Launch Streamlit App
streamlit run app/app.py


(Optional: Start your LLM backend like Ollama or HuggingFace Inference API.)

### Supported Datasets
Dataset	Type	Notes
PlantVillage	Controlled Lab	Multi-crop, >50 classes
PlantDoc	Field	Noisy real-world images
Cassava Leaf Disease	Field	5 classes
Rice Leaf Disease	Field	6 classes

Each dataset is documented in:

data/dataset_cards.md

### Model Summary
Stage	Model	Output
Leaf Detection	YOLO	Bounding box
Augmentation	Albumentations	Robust training images
Species Classification	ViT-Base	Species label
Disease Classification	ViT-Base	Disease label
Retrieval	MiniLM + FAISS	Relevant text
Conversational Layer	Mistral / TinyLLaMA	Actionable guidance

Safety & Ethical Use

This platform provides decision support, not certified diagnosis

Always consult agronomists for pesticide and chemical treatments

Avoid storing personal identifiable data

The LLM is tuned to be factual, neutral, and non-prescriptive

Follow all local regulations for crop protection products

📚 Citation

If using this project, please cite:

Deysel, D. (2025). Crop Disease Detection — From Lab to Field.
University of Miami, Principles & Practices of AI.
