# Loan Qualification 🔎💳
Automate loan-eligibility decisions from bank account data with a clean ML pipeline and a Streamlit app.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## Overview
This project builds an end-to-end workflow to determine whether an individual qualifies for a loan based on account information. It includes:
- Data prep & feature engineering
- Model training and export to a portable pipeline (`loan_pipeline.pkl`)
- An interactive Streamlit front-end for single-case scoring

**Repo highlights**: `app_streamlit.py`, `train_and_export.py`, `requirements.txt`, `loan_pipeline.pkl`, `schema.json`, `notebook/`, `data/input/`. :contentReference[oaicite:0]{index=0}

---

## Compare models steps
- Fill missing value with KNN Impputer
- Normalize our data
- Fit our data with LogisticRegression, KNN, and ANN models
- Compare model's accuracy with scaled and non-scaled data
- Use GridsearchCV models (Decision Tree/RandomForestClassifier/GradientBoostingClassifier/...)
- After all, compare the best models
  
---

## Create Application
Use RandomForestClassifier model to predict loan status in application
**Two Types of Data should be enter by user:**
- Categorical
- Numerical
  
---
## Quickstart

### Clone & set up
```bash
git clone https://github.com/Soufiiani/Loan_Qualification.git
cd Loan_Qualification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
