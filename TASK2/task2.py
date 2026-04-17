import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f5f0e6;
    color: black;
}
h1, h2, h3, p, label {
    color: black !important;
}
.stButton>button {
    background-color: #8b5e3c;
    color: white;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "model_simple.pkl"
SCALER_FILE = "scaler_simple.pkl"

@st.cache_resource
def train_model():
    df = pd.read_csv("fraudTrain.csv").dropna()

    df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

    df["gender"] = df["gender"].map({"M": 0, "F": 1})
    
    category_map = {cat: i for i, cat in enumerate(df["category"].unique())}
    df["category"] = df["category"].map(category_map)

    features = ["amt", "city_pop", "gender", "category", "hour"]
    X = df[features]
    y = df["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        class_weight={0:1, 1:10},
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, scaler

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
else:
    with st.spinner("Setting up system..."):
        model, scaler = train_model()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check if it is safe.")
st.write("Made by Jiya Gupta")

st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount", 0.0, 100000.0, 100.0)
city_pop = st.number_input("City Population", 0, 10000000, 50000)
gender = st.selectbox("Gender", ["Male", "Female"])
category = st.selectbox("Category", ["Food", "Shopping", "Travel", "Bills", "Other"])
hour = st.slider("Transaction Hour", 0, 23, 12)


gender_val = 0 if gender == "Male" else 1
category_val = ["Food", "Shopping", "Travel", "Bills", "Other"].index(category)

input_data = np.array([[amount, city_pop, gender_val, category_val, hour]])

input_scaled = scaler.transform(input_data)

if st.button("Check Transaction"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error("This transaction is likely fraudulent.")
    else:
        st.success("This transaction appears safe.")

    st.write("Risk Score:", round(prob, 2))


st.markdown("---")
st.write("Simple machine learning fraud detection system.")
