# 📰 Fake News Detector (SVM Model)

An intelligent **Fake News Detection system** that analyzes news articles and predicts whether they are 🟩 *Real* or 🟥 *Fake*.  
This project uses **Machine Learning (SVM)** with **TF-IDF Vectorization**, advanced **text preprocessing (SpaCy + NLTK)**, and a **modern Gradio UI** to provide fast, reliable, and interactive results.

---

## 🚀 Project Overview

In today’s digital world, misinformation spreads rapidly across the internet.  
The **Fake News Detector** leverages **Natural Language Processing (NLP)** and **Machine Learning** to identify fake news based on linguistic and semantic patterns in the text.

### 🎯 Goal:
To create a system that can automatically classify any news text as **Real** or **Fake**, and provide a **confidence level** of the prediction.

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Machine Learning Model** | Support Vector Machine (SVM) |
| **Feature Extraction** | TF-IDF (Term Frequency–Inverse Document Frequency) |
| **NLP Tools** | SpaCy, NLTK |
| **Interface** | Gradio |
| **Model Saving** | Joblib |
| **Data Handling** | Pandas |

---

## ⚙️ How It Works

1. **Data Loading:**  
   Two datasets — one containing *fake news* and the other containing *real news* — are combined and labeled.

2. **Text Cleaning & Lemmatization:**  
   - Lowercasing and punctuation removal  
   - Stopword removal (NLTK)  
   - Lemmatization (SpaCy)

3. **Feature Extraction:**  
   The cleaned text is converted into numerical features using **TF-IDF Vectorization**.

4. **Model Training:**  
   A **Linear SVM classifier** is trained on the processed data to distinguish between real and fake news.

5. **Evaluation:**  
   Model performance is measured using **Accuracy** and **F1-score**.

6. **Interactive UI (Gradio):**  
   The user inputs any news text and receives:
   - A **prediction** (🟩 Real News / 🟥 Fake News)
   - A **confidence level**
   - (Optional) Visualization and explanation of model confidence

---

