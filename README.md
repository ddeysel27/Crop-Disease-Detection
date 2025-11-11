🌾 Crop Disease Detection — From Lab to Field

Framing Statement (Assignment 6)
Empowering smallholder farmers and agricultural advisors with AI that detects crop diseases early from leaf images using vision transformers and generative retrieval-augmented chat, enabling faster diagnosis, evidence-based treatment, and reduced yield losses.

🧭 Overview

Crop Disease Detection is an end-to-end AI system combining computer vision and language models to analyze crop leaf images, identify likely diseases, and provide retrieval-grounded treatment advice in natural language.

The workflow integrates:

Segmentation (SAM 2 / U²-Net) → isolate leaf region

Data Augmentation (Albumentations) → improve robustness

Classification (Vision Transformer) → predict disease type

Explainability (Grad-CAM) → visualize attention regions

Calibration (Temperature Scaling) → realistic probabilities

Knowledge Retrieval (FAISS + Sentence-Transformers) → fetch relevant agronomy facts

Conversational LLM (Tiny LLaMA / Mistral) → summarize results into actionable guidance

⚙️ Quickstart

Create environment

python -m venv .venv
source .venv/Scripts/activate       # Windows
# or
source .venv/bin/activate           # macOS/Linux
pip install -r requirements.txt


Prepare dataset

Place source data inside data/raw/

Run

jupyter notebook notebooks/01_prepare_splits_preprocess.ipynb


→ writes segmented and preprocessed splits to data/processed/train|val|test/

Train

python -m src.models.train \
    --data data/processed \
    --model vit_base_patch16_224 \
    --epochs 15 --img 224


Evaluate
Open notebooks/03_evaluate_report.ipynb
→ View metrics, confusion matrix, Grad-CAM attention maps, and calibration plots.

Launch App

python app/gradio_app.py


Optionally start a local Ollama or Hugging Face endpoint for the conversational LLM layer.

🌱 Example Datasets
Dataset	Type	Source / License
PlantVillage	Controlled lab	Kaggle / CC BY-SA
PlantDoc	Field images	GitHub / CC BY
Cassava Leaf Disease	Field	TensorFlow DS
Rice Leaf Disease	Field	Mendeley

Maintain a data/dataset_cards.md summarizing:

Source & license

Number of classes & samples

Train/val/test split ratio

Any preprocessing or filtering notes

🧠 Model Lineup Summary
Stage	Model	Library	Output
🟢 Segmentation	SAM 2 / U²-Net	Meta / HF	Binary mask (leaf only)
🟡 Augmentation	Albumentations	albumentations	Robust images
🔵 Classifier	ViT-Base-Patch16	timm / transformers	Disease label + confidence
🟤 Explainability	Grad-CAM / Rollout	torchcam / custom	Heatmap overlay
🧩 Calibration	Temperature Scaling	netcal	Calibrated probabilities
🟠 Knowledge Retrieval	all-MiniLM-L6-v2 + FAISS	sentence-transformers	Vector index of agronomy text
🧠 Conversational LLM	Tiny LLaMA / Mistral	Ollama / HF	Farmer-friendly advice
⚙️ Deployment	Gradio Blocks	gradio	Unified web app
🔒 Safety & Ethical Use

This system provides decision support, not autonomous diagnosis.

Always verify results with qualified agronomists.

Respect pesticide and biosecurity regulations.

Avoid storing personally identifiable information or location data.

The conversational module is fine-tuned for factual, neutral advice — not prescriptive treatment.

📚 Citation / Acknowledgment

If you use this project or adapt its architecture, please cite:

Deysel, D. (2025). Crop Disease Detection — From Lab to Field.
University of Miami, Principles & Practices of AI 