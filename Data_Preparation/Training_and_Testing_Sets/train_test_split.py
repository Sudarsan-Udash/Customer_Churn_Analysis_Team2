# -*- coding: utf-8 -*-
"""
================================================================================
DATA PREPARATION — STEP 2: TRAIN / TEST SPLIT
================================================================================
Project : Customer Churn Analytics & Revenue Recovery
Outputs : X_train.csv, X_test.csv, y_train.csv, y_test.csv
          split_summary.txt
================================================================================
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE = '/content/drive/MyDrive/Colab Notebooks/Data_Preparation/'

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATASET
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE + 'preprocessed_dataset.csv')
print(f"Loaded preprocessed dataset: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFINE FEATURES AND TARGET
# ─────────────────────────────────────────────────────────────────────────────
# Features used: tenure and MonthlyCharges (the two primary business drivers)
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn_numeric']

# ─────────────────────────────────────────────────────────────────────────────
# 3. APPLY STANDARDSCALER
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=['tenure_scaled', 'MonthlyCharges_scaled'])

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT: 80% TRAIN / 20% TEST
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE SPLIT FILES
# ─────────────────────────────────────────────────────────────────────────────
X_train.to_csv(BASE + 'X_train.csv', index=False)
X_test.to_csv(BASE  + 'X_test.csv',  index=False)
y_train.to_csv(BASE + 'y_train.csv', index=False)
y_test.to_csv(BASE  + 'y_test.csv',  index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATE SPLIT SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
churn_train = y_train.sum()
churn_test  = y_test.sum()

summary = f"""
================================================================================
TRAIN / TEST SPLIT SUMMARY
================================================================================
Author          : Saurav Kumar
Project         : Customer Churn Analytics & Revenue Recovery
Split Strategy  : train_test_split | test_size=0.2 | random_state=42
Scaling Applied : StandardScaler (mean=0, std=1)
Features Used   : tenure_scaled, MonthlyCharges_scaled
--------------------------------------------------------------------------------

DATASET SIZES
  Total customers    : {len(df)}
  Training set       : {X_train.shape[0]} rows ({X_train.shape[0]/len(df)*100:.1f}%)
  Testing set        : {X_test.shape[0]} rows  ({X_test.shape[0]/len(df)*100:.1f}%)

CLASS COMPOSITION (Churn = 1)
  Churned in train   : {churn_train} ({churn_train/len(y_train)*100:.1f}% of train)
  Churned in test    : {churn_test}  ({churn_test/len(y_test)*100:.1f}% of test)
  Stayed   in train  : {len(y_train)-churn_train} ({(len(y_train)-churn_train)/len(y_train)*100:.1f}% of train)
  Stayed   in test   : {len(y_test)-churn_test}  ({(len(y_test)-churn_test)/len(y_test)*100:.1f}% of test)

FILES SAVED
  X_train.csv        — Scaled training features
  X_test.csv         — Scaled testing features
  y_train.csv        — Training labels (Churn_numeric)
  y_test.csv         — Testing labels  (Churn_numeric)

NOTES
  The 80/20 split was chosen to maximise training data while keeping a
  meaningful held-out evaluation set. The random_state=42 ensures full
  reproducibility across all team members and runs.
================================================================================
"""

print(summary)

with open(BASE + 'split_summary.txt', 'w') as f:
    f.write(summary)

print("All split files saved to Data_Preparation folder.")
