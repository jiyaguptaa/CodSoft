import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #f5f0e6;  /* beige */
        color: black;
    }

    /* Text styling */
    h1, h2, h3, h4, p, label {
        color: black;
        text-align: center;
    }

    /* Input box */
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
        border-radius: 6px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #8b5e3c;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }

    .stButton>button:hover {
        background-color: #6f4e37;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Spam SMS Detection")
st.markdown("<h3>Made by Jiya Gupta</h3>", unsafe_allow_html=True)

df = pd.read_csv("spam.csv", encoding='latin-1')

if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

user_input = st.text_input("Enter your SMS message")

if st.button("Check Message"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("This message is classified as SPAM")
        else:
            st.success("This message is classified as LEGITIMATE")
    else:
        st.warning("Please enter a message")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Machine Learning Project</p>", unsafe_allow_html=True)
