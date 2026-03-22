# -*- coding: utf-8 -*-
"""
================================================================================
DATA PREPARATION — STEP 1: PREPROCESSING & SAVING PREPROCESSED DATASET
================================================================================
Project : Customer Churn Analytics & Revenue Recovery
Output  : preprocessed_dataset.csv
================================================================================
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
filepath = '/content/drive/MyDrive/Colab Notebooks/Dataset_ATS_v2.csv'
data = pd.read_csv(filepath)

print("─" * 60)
print("RAW DATA AUDIT")
print("─" * 60)
print(f"Shape          : {data.shape}")
print(f"Missing Values :\n{data.isnull().sum()}")
print(f"\nTarget Distribution:\n{data['Churn'].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
data.fillna(method='ffill', inplace=True)
print("\nMissing values handled via forward-fill.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ENCODE TARGET VARIABLE: Churn → 1 (Yes), 0 (No)
# ─────────────────────────────────────────────────────────────────────────────
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data['Churn_numeric'] = data['Churn']

# ─────────────────────────────────────────────────────────────────────────────
# 4. BINARY ENCODE YES/NO AND GENDER COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
binary_cols = ['gender', 'Dependents', 'PhoneService', 'MultipleLines']
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

# ─────────────────────────────────────────────────────────────────────────────
# 5. ONE-HOT ENCODE MULTI-CATEGORY COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
df = pd.get_dummies(data, columns=['InternetService', 'Contract'], drop_first=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df['CustomerValue']    = df['tenure'] * df['MonthlyCharges']
df['TotalServices']    = df[['PhoneService', 'MultipleLines']].sum(axis=1)

if 'InternetService_Fiber optic' in df.columns:
    df['Senior_Fiber_Risk'] = (
        (df['SeniorCitizen'] == 1) & (df['InternetService_Fiber optic'] == 1)
    ).astype(int)

df['ContractRisk'] = np.select(
    [df['Contract_Two year'] == 1, df['Contract_One year'] == 1],
    [0, 1], default=2
)

df = df.drop(columns=['tenure_group'], errors='ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE PREPROCESSED DATASET
# ─────────────────────────────────────────────────────────────────────────────
output_path = '/content/drive/MyDrive/Colab Notebooks/Data_Preparation/preprocessed_dataset.csv'
df.to_csv(output_path, index=False)

print("\n─" * 60)
print("PREPROCESSED DATASET SAVED")
print("─" * 60)
print(f"Output path    : {output_path}")
print(f"Final shape    : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"\nClass balance  :\n{df['Churn'].value_counts()}")
print(f"\nPreview:")
display(df.head())
