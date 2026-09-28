# 🤖 FAQ Assistant — CodeAlpha Internship Task 2

An NLP-based FAQ chatbot that matches a user's natural-language question to the most relevant FAQ using **NLTK, TF-IDF, and Cosine Similarity**.

## Features

- FAQ knowledge base with categories
- Text preprocessing using NLTK
- Stopword removal
- Lemmatization
- TF-IDF vectorization
- Cosine similarity matching
- Similarity/confidence display
- Low-confidence fallback response
- Chat history
- Example questions
- Clear chat button
- Streamlit web interface

## Tech Stack

- Python
- NLTK
- Scikit-learn
- Streamlit

## Project Structure

```text
faq-chatbot/
├── app.py
├── faq_data.json
├── requirements.txt
└── README.md
```

## Setup

### 1. Open the project folder

```bash
cd faq-chatbot
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

The browser will open the chatbot, normally at:

```text
http://localhost:8501
```

## How it works

```text
User Question
      ↓
Text Preprocessing
      ↓
Tokenization + Stopword Removal + Lemmatization
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Best FAQ Match
      ↓
Answer
```

## Internship Task Mapping

| Requirement | Implementation |
|---|---|
| Collect FAQs | `faq_data.json` |
| Preprocess text | NLTK |
| Match similar FAQ | TF-IDF + Cosine Similarity |
| Display answer | Streamlit chat UI |
| Simple chat UI | Streamlit |

## Customization

Replace the questions and answers in `faq_data.json` with FAQs for your chosen domain. You can use college, banking, e-commerce, healthcare, travel, or company-support FAQs.

## Important

This is a retrieval-based FAQ chatbot. It does not generate new answers with an LLM. It selects the closest answer from the existing FAQ knowledge base.
