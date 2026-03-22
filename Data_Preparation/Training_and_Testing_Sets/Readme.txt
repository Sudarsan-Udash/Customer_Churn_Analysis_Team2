TRAIN / TEST SPLIT SUMMARY

Author          : Team 2
Project         : Customer Churn Analytics & Revenue Recovery
Split Strategy  : train_test_split | test_size=0.2 | random_state=42
Scaling Applied : StandardScaler (mean=0, std=1)
Features Used   : tenure_scaled, MonthlyCharges_scaled
--------------------------------------------------------------------------------

DATASET SIZES
  Total customers    : 7043
  Training set       : 5634 rows (80.0%)
  Testing set        : 1409 rows (20.0%)

CLASS COMPOSITION (Churn = 1)
  Churned in train   : 1496 (26.6% of train)
  Churned in test    : 373  (26.5% of test)
  Stayed   in train  : 4138 (73.4% of train)
  Stayed   in test   : 1036 (73.5% of test)

FILES SAVED
  X_train.csv        — Scaled training features
  X_test.csv         — Scaled testing features
  y_train.csv        — Training labels (Churn_numeric)
  y_test.csv         — Testing labels  (Churn_numeric)

NOTES
  The 80/20 split was chosen to maximise training data while keeping a
  meaningful held-out evaluation set. The random_state=42 ensures full
  reproducibility across all team members and runs.
