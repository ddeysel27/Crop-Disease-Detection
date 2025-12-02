# 🌱 Crop Disease Detection — End-to-End AI System

An advanced multi-model AI pipeline combining **Vision Transformers (ViT)** for plant species + disease classification, **LLM-powered agronomy assistance**, and a **RAG knowledge retrieval system**, deployed in a clean and interactive **Streamlit** application.

Users can upload a leaf image and instantly receive:

- ✔ **Predicted plant species**
- ✔ **Predicted disease class** (auto-routed to the correct model)
- ✔ **Confidence scores**
- ✔ **LLM-generated biological explanation + treatment guidance**
- ✔ **A RAG chatbot grounded in scientific literature**

---

## 🚀 Demo Video
## 🎥 Demo Video
[![Demo Video]](https://www.youtube.com/watch?v=https://youtu.be/6DOHdqP1464)

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

# 📁 Project Structure

CROP-DISEASE-DETECTION/
├── .venv/
├── app/
│   ├── pages/
│   │   ├── 00_Welcome_Page.py
│   │   ├── 01_Supported_Species_Info.py
│   │   ├── 02_Upload_and_Classify.py
│   │   ├── 03_Talk_to_Our_Chatbot.py
│   │
│   ├── species_images/
│   │   ├── apple.JPG
│   │   ├── blueberry.JPG
│   │   ├── cassava.jpg
│   │   ├── cherry.JPG
│   │   ├── corn.JPG
│   │   ├── grape.JPG
│   │   ├── milkdew.JPG
│   │   ├── orange.JPG
│   │   ├── peach.JPG
│   │   ├── pepperbell.JPG
│   │   ├── potato.JPG
│   │   ├── raspberry.JPG
│   │   ├── rice.jpg
│   │   ├── soybean.JPG
│   │   ├── squash.JPG
│   │   ├── strawberry.JPG
│   │   ├── tomato.JPG
│   │
│   ├── utils/
│       ├── detection/
│       │   ├── leaf_detector.py
│       │
│       ├── explainability/
│       │   ├── gradcam_pp.py
│       │
│       ├── load_model/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │
│       ├── ood/
│       │   ├── ood_detector.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── inference.py
│       │
│       ├── visuals/
│       │   ├── radar_plot.py
│       │   ├── uncertainty_badge.py
│       │
│       ├── llm_client.py
│       ├── model_utils.py
│       ├── species_mapping.py
│       ├── app.py
│       ├── config.py
│
├── data/
│   ├── processed/
│   │   ├── Cassava/
│   │   ├── PlantDoc/
│   │   ├── PlantVillage/
│   │   ├── Rice/
│   │   ├── Species_split/
│   │   ├── PlantDoc.zip
│   │   ├── species_split.zip
│   │
│   ├── raw/
│   │   ├── archive/
│   │   ├── cassava-leaf-disease-classification/
│   │   ├── PlantDoc_repo_new/
│   │   ├── rice-leaf-disease/
│   │   ├── processed.zip
│   │   ├── dataset_cards.md
│   │   ├── welcome_page.jpg
│
├── models/
│   ├── cassava_best.pth
│   ├── plant_village_best.pth
│   ├── rice_leaf_best.pth
│   ├── species_classifier_vit.pth
│   ├── yolo_plantdoc_detect.pt
│
├── notebooks/
│   ├── 00_explore_dataset_cassava.ipynb
│   ├── 00_explore_dataset_plantdoc.ipynb
│   ├── 00_explore_dataset_plantVillage.ipynb
│   ├── 00_explore_dataset_riceleafs.ipynb
│   ├── 01_prepare_cassava.ipynb
│   ├── 01_prepare_plantdoc.ipynb
│   ├── 01_prepare_plantVillage.ipynb
│   ├── 01_prepare_riceleaf.ipynb
│   ├── 01_prepare_splits_preprocess.ipynb
│   ├── 02_train_cassava.ipynb
│   ├── 02_train_classifier.ipynb
│   ├── 02_train_plantdoc.ipynb
│   ├── 02_train_plantVillage.ipynb
│   ├── 02_train_riceleafs.ipynb
│   ├── SPECIES_classifier.ipynb
│
├── scripts/
│   ├── prep_test_images.py
│
├── tests/
│   ├── test_images/
│   ├── batch_results.csv
│   ├── evaluate_model.ipynb
│   ├── inference_pipeline.ipynb
│   ├── test_classifiers.ipynb
│
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── latest_result.txt
└── requirements.txt



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
