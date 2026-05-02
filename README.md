# Word Similarity Project (NLP)

## 📌 Overview

This project builds a word similarity system step by step, starting from a simple context-based approach and gradually improving it using more advanced NLP techniques.

The goal is to understand how machines represent word meanings numerically and how similarity between words can be computed.

---

## 🚀 Project Structure

```
word-similarity-project/
│
├── data/
│   ├── data.txt
│   └── big_data.txt
│
├── 01_basic_context.py
├── 02_tfidf.py
├── 03_tfidf_improved.py
├── 04_word2vec.py
├── 05_word2vec_with_big_data.py
│
├── README.md
└── requirements.txt
```

---

## 🧠 Development Process

### 1. Basic Context-Based Similarity

* Implemented word similarity using co-occurrence context
* Built word vectors manually using frequency counts
* Used cosine similarity to compare words

---

### 2. TF-IDF Vectorization

* Applied TF-IDF to represent words more effectively
* Reduced the influence of common words (e.g., "is", "the")
* Improved similarity accuracy compared to the basic model

---

### 3. Improved TF-IDF Model

* Added preprocessing steps such as:

  * Stopwords removal
  * Text cleaning (punctuation removal)
* Adjusted context window size
* Achieved more meaningful similarity results

---

### 4. Word2Vec (Small Dataset)

* Introduced neural word embeddings
* Learned word representations based on surrounding context
* Observed semantic relationships between words

---

### 5. Word2Vec with Large Dataset (WikiText)

* Scaled the model using a larger text corpus
* Improved generalization and semantic quality
* Captured deeper relationships between words

---

## 🔍 Key Concepts

* **Distributional Hypothesis**
  ("Words that appear in similar contexts have similar meanings")

* Word Vector Representation

* Cosine Similarity

* TF-IDF

* Word Embeddings (Word2Vec)

---

## 📊 Example Results

```
king → queen, prince, monarch  
cat → dog, animal, pet  
city → town, village, urban  
```

---

## 🛠️ Tech Stack

* Python
* Gensim
* Scikit-learn
* Hugging Face Datasets

---

## 💡 What I Learned

* How to convert text into numerical representations
* Importance of preprocessing in NLP
* Differences between rule-based and learning-based methods
* How dataset size impacts model performance
* Practical understanding of word embeddings

---

## 🔧 How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run any script:

```
python 05_word2vec_with_big_data.py
```

---

## 📈 Future Improvements

* Add visualization (PCA / t-SNE)
* Extend to Korean NLP using KoNLPy
* Experiment with FastText or BERT
* Build a simple UI for interaction

---
