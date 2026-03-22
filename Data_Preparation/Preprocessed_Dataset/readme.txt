
PREPROCESSED DATASET SUMMARY

Project         : Customer Churn Analytics & Revenue Recovery
Step            : Data Preparation — Preprocessing
Output File     : preprocessed_dataset.csv


ORIGINAL DATASET
  Total records       : 7043
  Initial features    : 21

PREPROCESSING STEPS
  1. Missing Values Handling
     - Forward-fill method applied to handle missing data

  2. Target Encoding
     - Churn converted to numeric:
         Yes → 1
         No  → 0
     - New column created: Churn_numeric

  3. Binary Encoding
     - Converted categorical variables to numeric:
         gender (Female=0, Male=1)
         Dependents (No=0, Yes=1)
         PhoneService (No=0, Yes=1)
         MultipleLines (No=0, Yes=1)

  4. One-Hot Encoding
     - Applied to:
         InternetService
         Contract
     - Drop-first applied to avoid multicollinearity

  5. Feature Engineering
     - CustomerValue     = tenure × MonthlyCharges
     - TotalServices     = PhoneService + MultipleLines
     - Senior_Fiber_Risk = SeniorCitizen & Fiber optic users
     - ContractRisk      = Risk level based on contract type

  6. Feature Cleanup
     - Removed column: tenure_group (if present)

FINAL DATASET
  Total records       : 7043
  Total features      : 16

TARGET DISTRIBUTION (Churn)
  Churn = 1 (Yes)     : 1869 (~26.5%)
  Churn = 0 (No)      : 5174 (~73.5%)



NOTES
  - Dataset is fully numeric and ready for machine learning
  - No missing values remain after preprocessing
  - Features engineered to improve predictive performance
