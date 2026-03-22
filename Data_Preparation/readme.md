
# Data Preparation Module

## 📌 Overview
This module handles:
- Data cleaning
- Feature engineering
- Scaling
- Dataset splitting

---

## 📂 Files

### preprocess_dataset.py
- Loads raw dataset
- Cleans and encodes data
- Performs feature engineering
- Outputs:
  - preprocessed_dataset.csv

---

### train_test_split.py
- Loads preprocessed dataset
- Applies StandardScaler
- Splits dataset (80/20)
- Outputs:
  - X_train.csv
  - X_test.csv
  - y_train.csv
  - y_test.csv
  - split_summary.txt

---

### Scaling_Techniques_Documentation.docx
- Explains:
  - Why scaling is required
  - Z-score formula
  - Before vs after comparison
  - Code implementation

---

## 📊 Key Concept: Scaling
StandardScaler is used to normalise features:
- tenure
- MonthlyCharges

This ensures fair clustering performance.

---

## ▶️ Execution
Run:
