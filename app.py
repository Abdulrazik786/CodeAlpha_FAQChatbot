
import json
import re
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: .5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .small-muted {
        color: #64748b;
        font-size: .85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# NLTK resources
# -----------------------------
@st.cache_resource
def prepare_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

prepare_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_faqs():
    file_path = Path(__file__).with_name("faq_data.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

faqs = load_faqs()

# -----------------------------
# NLP preprocessing
# -----------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]
    return " ".join(tokens)

processed_questions = [preprocess_text(x["question"]) for x in faqs]

@st.cache_resource
def build_vectorizer(questions):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

vectorizer, faq_matrix = build_vectorizer(tuple(processed_questions))

# -----------------------------
# Matching engine
# -----------------------------
def find_best_match(user_question: str):
    cleaned = preprocess_text(user_question)

    if not cleaned.strip():
        return None, 0.0

    query_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(query_vector, faq_matrix)[0]

    best_index = int(scores.argmax())
    return faqs[best_index], float(scores[best_index])

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🤖 FAQ Assistant")
    st.caption("NLP-powered FAQ matching")

    st.markdown("---")
    st.markdown("### How it works")
    st.write("1. Preprocess the question")
    st.write("2. Convert text to TF-IDF vectors")
    st.write("3. Calculate cosine similarity")
    st.write("4. Return the closest FAQ answer")

    st.markdown("---")
    st.markdown("### Knowledge Base")
    st.metric("FAQs", len(faqs))
    st.metric("Categories", len(set(x["category"] for x in faqs)))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Main header
# -----------------------------
st.markdown('<div class="main-title">🤖 FAQ Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask a question in natural language and get the most relevant answer from the FAQ knowledge base.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Example questions
# -----------------------------
st.markdown("#### Try asking")
examples = [
    "How can I reset my password?",
    "Where is my order?",
    "What payment methods can I use?",
    "How do I return an item?",
]

cols = st.columns(4)
for i, example in enumerate(examples):
    with cols[i]:
        if st.button(example, use_container_width=True, key=f"example_{i}"):
            st.session_state.pending_question = example

# -----------------------------
# Chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

# -----------------------------
# Input
# -----------------------------
typed_question = st.chat_input("Ask your question...")
pending_question = st.session_state.pop("pending_question", None)
user_question = typed_question or pending_question

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    faq, score = find_best_match(user_question)

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if faq is None:
            answer = "Please enter a question so I can help you."
            st.markdown(answer)
        elif score < 0.25:
            answer = (
                "I couldn't find a sufficiently relevant FAQ for that question. "
                "Please try rephrasing it or contact customer support."
            )
            st.markdown(answer)
            st.caption(f"Confidence: {score:.0%} · Match threshold: 25%")
        else:
            answer = faq["answer"]
            st.markdown(f"**{answer}**")
            st.caption(
                f"Category: {faq['category']} · "
                f"Confidence: {score:.0%} · "
                f"Matched FAQ: {faq['question']}"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": (
            f"Category: {faq['category']} · Confidence: {score:.0%}"
            if faq and score >= 0.25
            else f"Confidence: {score:.0%}"
        )
    })

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    '<div class="small-muted">Built with Python • NLTK • TF-IDF • Cosine Similarity • Streamlit</div>',
    unsafe_allow_html=True
)
