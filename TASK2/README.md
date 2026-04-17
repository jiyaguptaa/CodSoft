# Task 2

## Overview
Credit Card Fraud Detection web app using Streamlit.

This project trains a `RandomForestClassifier` on transaction data and offers an interactive interface to predict whether a transaction is likely fraudulent. It also caches trained artifacts to speed up subsequent runs.

## What it does

- Loads `fraudTrain.csv`
- Extracts features: `amt`, `city_pop`, `gender`, `category`, `hour`
- Uses `StandardScaler` and `RandomForestClassifier`
- Saves/loads trained artifacts: `model_simple.pkl`, `scaler_simple.pkl`

## How to run

```powershell
cd TASK2
pip install pandas numpy scikit-learn streamlit joblib
streamlit run task2.py
```

## Notes

- The app trains a new model only if `model_simple.pkl` and `scaler_simple.pkl` do not exist.
- Transaction categories are mapped to numeric encoding and gender is converted to binary values.
- Keep `fraudTrain.csv` in the `TASK2` folder.
