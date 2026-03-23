import pandas as pd
import numpy as np
import re
import string
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

st.markdown("""
<style>
    .stApp {
        background-color: #f5f0e6;
        color: #0a0806;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #0a0806;
    }
    .subtitle {
        text-align: center;
        color: #0a0806;
        margin-bottom: 20px;
    }
    textarea {
        background-color: #fffaf3 !important;
        color: #0a0806 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    label {
        color: #0a0806 !important;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #d6bfa9;
        color: #0a0806;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #c4a484;
    }
    div[data-testid="stAlert"] {
        background-color: #e8dfd3 !important;
        color: #0a0806 !important;
        border-radius: 10px;
        border: none;
    }
    .footer {
        text-align: center;
        color: #0a0806;
        font-size: 14px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def advanced_clean(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_and_train():
    df = pd.read_csv(
        r"C:\Users\jiyag\OneDrive\Desktop\CodSoft\train_data.txt",
        sep=":::",
        engine="python",
        encoding="latin-1",
        names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
    )

    df["clean_text"] = df["DESCRIPTION"].apply(advanced_clean)
    df["GENRE"] = df["GENRE"].apply(lambda x: [g.strip() for g in x.split(",")])

    df = df.sample(8000, random_state=42)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["GENRE"])

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(df["clean_text"])

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X, y)

    return model, tfidf, mlb

model, tfidf, mlb = load_and_train()

def predict_genre(plot, threshold=0.3):
    cleaned = advanced_clean(plot)
    vec = tfidf.transform([cleaned])
    probs = model.predict_proba(vec)[0]

    results = []
    for i, p in enumerate(probs):
        if p > threshold:
            results.append((mlb.classes_[i], round(p, 2)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

st.markdown('<div class="title">Movie Genre Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a movie plot — I\'ll guess the genres! 🎬</div>', unsafe_allow_html=True)

user_input = st.text_area(
    "Paste or write the movie plot here...",
    height=220,
    key="plot_input"
)

threshold = st.slider(
    "Minimum confidence to show a genre",
    min_value=0.05,
    max_value=0.50,
    value=0.30,
    step=0.05,
    help="Lower the value if no genres appear (try 0.15–0.20 for most plots)"
)

st.markdown('<div class="footer">Made by Jiya Gupta</div>', unsafe_allow_html=True)

if st.button("Predict Genre", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a movie plot first 📝")
    else:
        with st.spinner("Analyzing your plot... 🎥"):
            predictions = predict_genre(user_input, threshold=threshold)

        if predictions:
            st.success("**Predicted Genres** (sorted by confidence):")
            
            for genre, conf in predictions[:6]:
                st.write(f"**{genre}**")
                st.progress(int(conf * 100))
                st.caption(f"{conf:.2f}")
            
            if len(predictions) > 6:
                with st.expander("Show more lower-confidence genres"):
                    for genre, conf in predictions[6:]:
                        st.write(f"{genre} ({conf})")
        else:
            st.info(f"No genres above {threshold:.2f} confidence. Try lowering the slider or using a more detailed plot.")

st.markdown("---")
st.caption("Movie Genre Classification • TF-IDF + Logistic Regression • 8000 training samples")