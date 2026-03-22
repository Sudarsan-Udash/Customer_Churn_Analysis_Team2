# -*- coding: utf-8 -*-
"""
================================================================================
CLUSTERING ANALYSIS — STEP 2: TRAIN K-MEANS MODEL & SAVE
================================================================================
Project : Customer Churn Analytics & Revenue Recovery
Outputs : kmeans_model.pkl, scaler.pkl, clustered_dataset.csv
================================================================================
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE       = '/content/drive/MyDrive/Colab Notebooks/'
DATA_PREP  = BASE + 'Data_Preparation/'
CLUST_OUT  = BASE + 'Clustering_Analysis/'

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PREP + 'preprocessed_dataset.csv')
print(f"Loaded: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCALE FEATURES (tenure + MonthlyCharges)
# ─────────────────────────────────────────────────────────────────────────────
X_cluster = df[['tenure', 'MonthlyCharges']]
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN K-MEANS MODEL (k=3, k-means++ initialisation)
# ─────────────────────────────────────────────────────────────────────────────
# k=3 selected via Elbow Method (see optimal_clusters_elbow.py)
# k-means++ ensures better centroid initialisation vs random
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nModel trained. Inertia (WCSS) = {kmeans.inertia_:,.2f}")
print(f"Cluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL PERSONAS
# ─────────────────────────────────────────────────────────────────────────────
persona_map  = {0: "Loyal High-Rollers", 1: "At-Risk Newbies", 2: "Stable Budgeteers"}
cluster_map  = {0: "The Loyal High-Rollers", 1: "The At-Risk Newbies", 2: "The Stable Budgeteers"}
df['Persona']      = df['Cluster'].map(persona_map)
df['Cluster_Name'] = df['Cluster'].map(cluster_map)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PERSONA FINANCIAL AUDIT
# ─────────────────────────────────────────────────────────────────────────────
persona_profile = df.groupby('Cluster_Name').agg({
    'tenure': 'mean',
    'MonthlyCharges': 'mean',
    'Churn_numeric': 'mean',
    'CustomerValue': 'sum'
}).rename(columns={
    'Churn_numeric': 'Churn Rate (%)',
    'CustomerValue': 'Total Revenue ($)'
})
persona_display = persona_profile.copy()
persona_display['Churn Rate (%)'] = (persona_display['Churn Rate (%)'] * 100).round(2).astype(str) + '%'
persona_display['Total Revenue ($)'] = persona_display['Total Revenue ($)'].map('${:,.2f}'.format)
print("\n--- PERSONA FINANCIAL PROFILES ---")
display(persona_display)

total_at_risk = df[df['Churn_numeric'] == 1]['CustomerValue'].sum()
print(f"\nTOTAL REVENUE AT RISK: ${total_at_risk:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE MODEL & SCALER AS PICKLE FILES
# ─────────────────────────────────────────────────────────────────────────────
with open(CLUST_OUT + 'kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("kmeans_model.pkl saved.")

with open(CLUST_OUT + 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("scaler.pkl saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE CLUSTERED DATASET
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(CLUST_OUT + 'clustered_dataset.csv', index=False)
print("clustered_dataset.csv saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE MODEL METADATA
# ─────────────────────────────────────────────────────────────────────────────
centers_real = scaler.inverse_transform(kmeans.cluster_centers_)
metadata = f"""
================================================================================
K-MEANS MODEL METADATA
================================================================================
Model           : KMeans (scikit-learn)
n_clusters      : 3
init            : k-means++
random_state    : 42
Features used   : tenure (scaled), MonthlyCharges (scaled)
Scaler          : StandardScaler
Final WCSS      : {kmeans.inertia_:,.2f}

CLUSTER CENTROIDS (real-world values after inverse transform)
  Cluster 0 — Loyal High-Rollers   : tenure={centers_real[0][0]:.1f} months, charges=${centers_real[0][1]:.2f}
  Cluster 1 — At-Risk Newbies      : tenure={centers_real[1][0]:.1f} months, charges=${centers_real[1][1]:.2f}
  Cluster 2 — Stable Budgeteers    : tenure={centers_real[2][0]:.1f} months, charges=${centers_real[2][1]:.2f}

FILES SAVED
  kmeans_model.pkl      — Trained KMeans model (load with pickle)
  scaler.pkl            — Fitted StandardScaler (use to transform new data)
  clustered_dataset.csv — Full dataset with Cluster, Persona, Cluster_Name columns

HOW TO LOAD AND USE
  import pickle
  with open('kmeans_model.pkl', 'rb') as f: kmeans = pickle.load(f)
  with open('scaler.pkl', 'rb') as f:       scaler = pickle.load(f)
  new_data_scaled  = scaler.transform(new_data[['tenure', 'MonthlyCharges']])
  cluster_labels   = kmeans.predict(new_data_scaled)
================================================================================
"""
print(metadata)
with open(CLUST_OUT + 'model_metadata.txt', 'w') as f:
    f.write(metadata)
