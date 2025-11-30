# 🗣️ Native Language Identification of Indian English Speakers Using HuBERT

This project develops an AI-based system that predicts a speaker’s **native Indian language** from their **English accent**, comparing traditional **MFCC features** with deep-learning-based **HuBERT embeddings**.  
A demo **accent-aware cuisine recommendation application** is also included.

---

## ⭐ Project Objectives
* Identify a speaker’s native language based on their English accent
* Compare **MFCC vs HuBERT** feature extraction methods
* Study **generalization across age groups** (train on adults, test on children)
* Compare **word-level vs sentence-level** accent recognition
* Perform **HuBERT layer-wise analysis** to determine the most informative layer
* Experiment with ML/DL models (**Random Forest, Logistic Regression, CNN, BiLSTM, Transformer-based**)
* Perform hyperparameter tuning and performance optimization
* Build an **Accent-Aware Cuisine Recommendation** system

---

## 📦 Dataset Used
* The project uses the **IndicAccentDb** dataset from Hugging Face  
  🔗 https://huggingface.co/datasets/DarshanaS/IndicAccentDb
* **This exact dataset was mounted on Google Drive and used in Google Colab** for model training, testing, and evaluation.
* Contains audio recordings of Indian speakers from multiple native languages:
  * **Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Odia, Gujarati, Marathi, Assamese**
* Includes:
  * **Adult vs Child** recordings (cross-age evaluation)
  * **Word-level and sentence-level** speech (linguistic-level evaluation)

---

## 📂 Repository Structure
├── app.py

├── notebooks/

| ├── 01_HuBERT_Feature_Extraction.ipynb
  
| ├── 02_HuBERT_Classification.ipynb
  
| ├── Cross_Age_Generalization.ipynb
  
| ├── HuBERT_Layerwise_Analysis.ipynb
  
| ├── Linguistic_Level_Generalization.ipynb
  
| └── MFCC_vs_HuBERT_Comparison.ipynb
  
├── models/

├── data/

├── images/

├── docs/

├── requirements.txt

└── README.md
---

## 🧠 Feature Extraction & Modeling

| Feature Type | Description |
|--------------|-------------|
| **MFCC** | Traditional handcrafted acoustic features |
| **HuBERT embeddings** | Self-supervised transformer-based deep speech representations |

### Models Explored
* Random Forest Classifier (final selected model)
* Logistic Regression
* CNN / BiLSTM / Transformer-based models

---

## 📊 Results Summary

| Method | Accuracy |
|--------|----------|
| MFCC + Random Forest | ~62% |
| **HuBERT + Random Forest** | **~87%** |

### Additional Findings
| Experiment | Outcome |
|-----------|---------|
| Adults → Children | Accuracy dropped from **~85%** to **~55%** |
| Word vs Sentence | **65% vs 87%** |
| Best HuBERT Layer | **Layer 7** |

---

## 🍽️ Real-World Application: Cuisine Recommendation System
Predicts accent → Infers region → Suggests traditional dishes

| Accent | Region | Recommended Dishes |
|--------|--------|--------------------|
| Malayalam-English | Kerala | Appam, Puttu, Avial |
| Hindi-English | North India | Chole Bhature, Aloo Paratha |
| Tamil-English | Tamil Nadu | Dosa, Idli, Sambar |

---

## 🛠 Tools & Frameworks
* Python
* **Google Colab**
* Hugging Face Transformers (HuBERT)
* Librosa
* Scikit-learn
* Pandas / NumPy
* Matplotlib / Seaborn
* Streamlit / Flask

---

# 🧪 Running the Project in Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -r requirements.txt
Open and run the notebooks in the /notebooks/ folder.
```


---
## 💻 Running Locally

Step 1 — Clone the Repository

git clone https://github.com/Navomy2020/Native-language-identification-of-Indian-english-speakers1.git

cd Native-language-identification-of-Indian-english-speakers1

Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Run the Streamlit Application
streamlit run app.py

# 📄 Conceptual Background
* An accent reflects pronunciation patterns influenced by a speaker’s native language (L1)
* Acoustic cues like vowel formation, consonant articulation, and prosody help identify native language
* HuBERT effectively encodes deep contextual accent features

---

# 🔮 Future Work
* Expand dataset to more Indian languages
* Improve performance for children’s speech
* Real-time microphone input & mobile deployment
* Multi-language UI support

---

# 👩‍💻 Team Members

* **Nandana Biju** :Research & Experimentation
* **Navomy Mariya Alex** :Model Training & App Development
* **Sulfa Saji** :Feature Engineering & Analysis
----- 
# 📄 License

For academic and research purposes only.
