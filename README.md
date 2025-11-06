#  AI Mental Doctor — Depression Detection Module

Experimental **AI Psychologist** that leverages **Natural Language Processing (NLP)** and **Deep Learning (RNN + Attention)** to analyze text conversations, detect **depression**, **anxiety**, and **emotional patterns**, and suggest **personalized self-care recommendations**.

##  Overview

This project explores the intersection of **Artificial Intelligence**, **Mental Health**, and **Language Understanding**.
It simulates an **AI-powered psychologist assistant** capable of identifying **emotional states** from written text.
The system combines **classical NLP models** and **deep neural architectures** to understand **subtle psychological cues** in language.

##  Motivation

Mental health conditions such as **depression** and **anxiety** affect hundreds of millions worldwide.  
Early detection and timely support are crucial, yet professional help is often **inaccessible or stigmatized**.

-  Over **400 million** people suffer from depression globally.  
-  Around **25% of Egyptians** face psychological disorders.  
-  Over **80%** of affected individuals in low- and middle-income regions receive **no treatment**.  

AI-based systems like this one can assist in **early awareness** and **guide individuals toward professional care** when needed.

---

##  Objectives

- Simulate **psychologist–patient reasoning** using AI.  
- Detect indicators of **depression**, **anxiety**, and **stress** from text.  
- Apply **Deep Learning** and **NLP** to identify **linguistic** and **emotional signals**.  
- Recommend **positive coping techniques** and **self-care strategies**.  
- Provide **modular, reproducible code** for further experimentation and research.  

---

##  System Design

###  Architecture

The project consists of three main pipelines:

1. **Preprocessing and Feature Extraction**  
   Cleans text, removes noise (emojis, links, symbols), tokenizes, and prepares embeddings.  

2. **Model Training**  
   - **Classical Machine Learning:** Logistic Regression, SVM, Random Forest.  
   - **Deep Learning:** Bidirectional LSTM with Attention mechanism.  

3. **Evaluation and Inference**  
   Computes **accuracy**, **F1 score**, **confusion matrices**, and **ROC curves**.  
   Provides **real-time inference** through a lightweight prediction service.  

---

###  Pipeline Summary

| **Stage**             | **Description** |
|------------------------|-----------------|
| **Data Cleaning**      | Removes noise and normalizes user text. |
| **Feature Engineering**| Converts text into TF-IDF and sequence embeddings. |
| **Model Training**     | Trains classical ML and BiLSTM+Attention models independently. |
| **Evaluation**         | Reports metrics and generates visualizations. |
| **Inference Service**  | Provides predictions for new inputs in real time. |


##  Key Features

- **Dual-model pipeline** — compares classical ML (SVM, Logistic Regression) vs. deep BiLSTM-Attention models.
- **Text preprocessing** — advanced cleaning, contraction handling, tokenization, and lemmatization.
- **Feature extraction** — TF-IDF and word-embedding representations (GloVe 100d).
- **Evaluation framework** — accuracy, F1, ROC-AUC, and visualization reports.
- **Explainability** — interpretable logistic features for depressive vs. non-depressive text.
- **Reproducible notebook** — modular, fully documented Jupyter workflow.

##  Methodology

### Preprocessing
- **Tokenization** and **lowercasing** with *NLTK* and *tweet-preprocessor*  
- **Stopword removal** and **lemmatization**  
- **Sequence conversion** and **padding** for deep models  

---

###  Classical ML Models
- **TF-IDF vectorization** for text representation  
- Models used: **Logistic Regression**, **SVM**, **Random Forest**  
- Model selection based on **F1-score** and **cross-validation**  

---

###  Deep Learning Model
- **Embedding Layer:** word vector representations (GloVe)  
- **Bidirectional LSTM:** captures forward and backward dependencies  
- **Attention Mechanism:** focuses on emotionally significant words  
- **Dense Output Layer:** performs binary classification (depressive vs. non-depressive)  
- **Optimizer:** Adam  
- **Loss Function:** Binary Cross-Entropy  

---

###  Evaluation Metrics
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **ROC-AUC**

---

###  Inference System
A lightweight **InferenceService** integrates preprocessing, tokenization, and prediction logic for both classical and deep models.  
It provides **instant predictions** for unseen user text.

---

###  Experimental Results
Example summary from **Colab execution**:
=== Final Summary ===
Classical (best) evaluation:
 - Accuracy: 0.9973
 - F1: 0.9885

Deep model evaluation:
 - Accuracy: 0.9893
 - F1: 0.9536

##  Model Performance (Sample)

| Model Type           | Accuracy | F1-Score | Notes             |
|----------------------|-----------|-----------|-------------------|
| **Linear SVC**        | 0.997     | 0.988     | Best classical     |
| **Logistic Regression** | 0.951     | 0.951     | Stable baseline    |
| **BiLSTM + Attention**  | 0.989     | 0.954     | Deep model generalization |

---

##  Example Insights

**Top words linked to depressive content:**  
“depression”, “hopeless”, “tired”, “empty”, “worthless”

**Top words linked to non-depressive content:**  
“happy”, “love”, “excited”, “amazing”, “smile”

---

##  Architecture

data/ → preprocessing → feature extraction
↓
┌─────────────────────────────┐
│ Classical Models (SVM, LR) │
└─────────────────────────────┘
│
▼
┌─────────────────────────────┐
│ Deep Models (BiLSTM + Attn)│
└─────────────────────────────┘
│
▼
Evaluation & Insights

---

##  Folder Structure

AI-Mental-Doctor/
│
├── code/
│ ├── improved_version/
│ │ └── AI_Mental_Doctor_Improved.ipynb
│ └── primitive_version/
│ └── AI_Mental_Doctor_Primitive.ipynb
│
├── data/
│ └── (sample or reference only – not uploaded)
│
├── results/
│ ├── logs/
│ ├── summaries/
│ └── plots/
│
├── docs/
│ └── (private research report not included)
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

## ⚙️ Installation & Usage

### 1. Clone the Repository
bash
git clone https://github.com/YOUR-USERNAME/AI-Mental-Doctor.git
cd AI-Mental-Doctor
### 2. Install Dependencies
bash
pip install -r requirements.txt
### 3. Run the Project (Google Colab Recommended)
Open main.ipynb in Google Colab or Jupyter Notebook, then execute cells sequentially (1–11).
All results (models, metrics, plots) will be saved under /project_results.

### 4. Inference Demo
examples = [
    "I feel hopeless and tired of everything.",
    "I had a really wonderful day and I'm grateful."
]
inference_service.predict(examples)

##  **Research Context**

This project forms part of a broader exploration of **AI for mental health analysis**.  
The work investigates how **linguistic signals** can reflect **psychological states**, while emphasizing **responsible use** and **ethical AI** in healthcare contexts.

Combines **Affective Computing** and **Computational Psychology** to explore AI-assisted emotional understanding.

The included research documents summarize:
- **Nora: The Empathetic Psychologist** — SVM/CNN-based emotional dialog agent
- **eRisk CLEF Challenge** — early depression detection via Reddit data
- **Prodromal Phase Detection** — identifying bipolar disorder through social media analysis
- **NRC Affect Intensity Lexicon** — emotion intensity mapping

This project extends these works by emphasizing **ethical awareness**, **non-patient applications**, and **contextual understanding**.

---

##  **Technologies Used**

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **NLP & ML** | scikit-learn, TensorFlow, Keras, NLTK |
| **Visualization** | Matplotlib, Seaborn |
| **Utilities** | tqdm, joblib, tweet-preprocessor |
| **Environment** | Jupyter Notebook / Google Colab |

---

##  **Limitations**

- Predictions are **text-based** and **not a substitute** for professional diagnosis.  
- Possible **linguistic or demographic biases** in datasets.  
- Emotional state inference is **probabilistic**.  
- Lacks **multimodal (audio/visual)** input support.

---

##  **Future Work**

- **Multimodal Emotion Detection** — integrate text, audio, and facial expressions.  
- **Context-Aware Dialogue** — transformer-based conversational memory.  
- **Explainable AI** — integrate SHAP/LIME for interpretability.  
- **Clinical Collaboration** — validation with professionals.  
- **Ethical Framework** — data privacy and informed consent mechanisms.  
- **Web & Mobile Interface** — front-end implementation using React or Flutter.
- Expand dataset with **multilingual** and **context-rich** inputs.  
- Integrate **emotion classification** (stress, anger, optimism).  
- Add **conversational interface** for interactive AI therapy simulation.  
- Fine-tune deep models using **transformer-based architectures** (BERT, RoBERTa).  
- Deploy **secure API** or **local inference** for privacy-preserving analysis.

---

##  License

This project’s **code** is released under the **MIT License**.  
All **reports, analyses, and documents** remain proprietary to the author  
and may not be reproduced or distributed without explicit permission.

---

## Author

**Maryam Kandil**  
Curious and driven **Software Engineer** exploring the intersection of **AI**, **Web Applications**, and **IoT Innovation**.  
Passionate about building **intelligent, human-centered systems** that connect technology with real-world well-being.
