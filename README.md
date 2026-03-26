# 🎵 MoodMate  
### Emotion-Aware Music Recommendation System

---

## Overview

**MoodMate** is an AI-powered system that detects a user's emotional state from text input and recommends music aligned with that emotion.

It integrates:

- Transformer-based NLP (**DistilBERT**)
- Deep learning classification
- Content-based music recommendation

---

## Features

- Emotion detection using a fine-tuned DistilBERT model  
- Handles class imbalance using weighted loss  
- Evaluation using accuracy, precision, recall, and F1-score  
- Music recommendation using TF-IDF + cosine similarity  
- Custom frontend (HTML/CSS/JS) with Streamlit backend  

---

## Model Architecture

**Base Model:** `distilbert-base-uncased`

**Custom Classification Head:**
- Linear (768 → 256)
- ReLU + Dropout
- Linear (256 → 128)
- ReLU + Dropout
- Linear (128 → 5)

**Loss Function:**
- CrossEntropyLoss with class weights

---

## Datasets

### 1. Emotion Dataset

- **Source:** Hugging Face (`boltuix/emotions-dataset`)
- **Type:** Supervised text classification  
- **Input:** Sentence / text  
- **Output:** Emotion label  

#### Original Classes
- Happiness, sadness, anger, fear, surprise, etc.

#### Final Emotion Classes

- Happy  
- Sad  
- Calm  
- Energetic  
- Angry  

#### Why Reduction?

- Simplifies classification  
- Improves generalization  
- Aligns with music recommendation categories  

---

### ⚙️ Preprocessing

- Tokenization using pretrained DistilBERT tokenizer  
- Automatic normalization (uncased)  
- Label mapping to reduced emotion set  

---

### Class Imbalance Handling

- Computed using `compute_class_weight()`  
- Applied in: CrossEntropyLoss(weight=class_weights)

  
---

###  Data Splitting

- 70% Training  
- 15% Validation  
- 15% Testing  
- Stratified splitting used  

---

## Music Dataset

- **Type:** Song metadata dataset  
- **Scale:** Thousands of songs (~35,000 tag combinations)

### Attributes

- Song name  
- Artist  
- Spotify track ID  
- Preview URL  
- Genre / mood tags  

---

## Data Preprocessing (Music)

- Converted tags to lowercase  
- Replaced underscores with spaces  
- Removed noisy/redundant tokens  
- Combined tags into `clean_tags`  
- Reduced 35,000+ combinations into meaningful groups  

---

## Tag Normalization

Mapped into categories:

- Rock, Metal, Pop, Hip-hop, EDM  
- Classical, Acoustic, Instrumental  
- Ambient, Lo-fi  
- Emotional tags: happy, sad, calm, energetic, angry  

---

## Emotion → Tag Mapping

| Emotion   | Tags                          |
|----------|------------------------------|
| Happy     | pop, happy                  |
| Sad       | acoustic, sad               |
| Calm      | classical, ambient, lo-fi   |
| Energetic | rock, hip-hop, edm          |
| Angry     | metal                       |

---

##  Feature Extraction

- **Technique:** TF-IDF  
- Converts tags → numerical vectors  
- Captures importance of tags  

---

##  Similarity Computation

- **Method:** Cosine Similarity  

Compares:
- Emotion query vector  
- Song tag vectors  

Top-K similar songs are recommended  

---

##  Training Details

- Batch size: 32  
- Epochs: 15  
- Learning rate: 1e-5  

### Evaluation Metrics

- Accuracy  
- Precision (Macro)  
- Recall (Macro)  
- F1 Score (Macro)  

---

##  Evaluation

- Confusion Matrix visualization  
- Classification Report  
- Training vs Validation loss plots  

---

##  Recommendation System

- Uses TF-IDF vectorization on song tags  
- Computes cosine similarity between:
- Emotion-based query  
- Song metadata  

---

##  Tech Stack

### Machine Learning
- Python  
- PyTorch  
- Hugging Face Transformers  
- Scikit-learn  

### Recommendation
- TF-IDF  
- Cosine Similarity  

### Frontend
- HTML, CSS, JavaScript  

### Backend
- Streamlit  

---

## 📁 Project Structure
MoodMate/
│

├── app.py

├── moodmate_pipeline.py

├── notebooks/

│ └── moodmate_db_nn.ipynb

├── requirements.txt

├── README.md


---

##  How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
<img width="1018" height="876" alt="Screenshot 2026-03-25 164628" src="https://github.com/user-attachments/assets/66a90563-ce50-4ec3-bef4-5635205dfcbb" />
