# Task 1

## Overview
Built a Movie Genre Classification System using Machine Learning � and turned it into an interactive web app. ??

The idea was simple: take a movie plot and predict its genre. Behind the scenes, the app uses TF-IDF for feature extraction and Logistic Regression for multi-label classification, wrapped in a clean Streamlit interface. ??

## What I learned

- Sampled a smaller training set for faster training without losing performance
- Added a confidence threshold to filter out weaker genre predictions
- Implemented text preprocessing (lowercasing, removing noise, punctuation removal), which improved accuracy

## Project details

- Dataset: `train_data.txt`
- Model: `LogisticRegression` via `OneVsRestClassifier`
- Feature extraction: `TfidfVectorizer`
- UI: `Streamlit`

## How to run

```powershell
cd TASK1
pip install pandas numpy scikit-learn streamlit
streamlit run task1.py
```

## Notes

- If `train_data.txt` path is not found, verify it exists in the `TASK1` folder.
- The app includes a slider for minimum confidence threshold and displays genre predictions sorted by confidence.

## GitHub
https://github.com/jiyaguptaa/CodSoft

#MachineLearning #Python #DataScience #Streamlit #AIProjects #StudentDeveloper #LearningByDoing #CodSoft #MLProject #Internship
