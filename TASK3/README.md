# Task 3


## Overview

Built a Spam SMS Detection System using Machine Learning and deployed it as an interactive web application.

The goal of this project is to classify SMS messages as spam or legitimate (ham). The system uses TF-IDF for feature extraction and a Multinomial Naive Bayes classifier to detect patterns commonly found in spam messages. The model is integrated into a minimal and user-friendly Streamlit interface.

## What I learned

- Understood how text data is transformed into numerical features using TF-IDF
- Learned how probabilistic models like Naive Bayes work for text classification
- Handled real-world dataset issues such as encoding errors and inconsistent column formats
- Improved UI/UX using custom styling in Streamlit (CSS overrides)
- Built a complete pipeline from data preprocessing → model training → deployment

## Challenges & Solutions

- Dataset inconsistency
    - Problem: Columns were labeled as v1, v2 instead of meaningful names
    - Solution: Renamed columns and mapped labels (spam → 1, ham → 0)
- Encoding issues while reading data
    - Problem: Errors while loading dataset
    - Solution: Used latin-1 encoding for compatibility
- UI styling not applying correctly
    - Problem: Background and theme not updating
    - Solution: Used .stApp selector instead of body for proper styling in Streamlit
- Model generalization
    - Problem: Low accuracy with small sample data
    - Solution: Switched to full dataset and proper train-test split

## Project Details

- Dataset: spam.csv (SMS Spam Collection Dataset)
- Model: MultinomialNB
- Feature Extraction: TfidfVectorizer
- Language: Python
- UI Framework: Streamlit

## How to Run

cd TASK3
pip install pandas numpy scikit-learn streamlit
streamlit run task3.py

## Features

- Real-time SMS classification (Spam / Legitimate)
- Clean and minimal beige-brown themed UI
- Fast prediction using trained model
- Displays model accuracy

## Notes

Ensure spam.csv is present in the TASK3 folder
If the dataset has columns like v1 and v2, the code automatically handles renaming
Encoding is set to latin-1 to avoid reading errors

## GitHub

https://github.com/jiyaguptaa/CodSoft

#MachineLearning #Python #DataScience #Streamlit #AIProjects #StudentDeveloper #LearningByDoing #CodSoft #MLProject #Internship