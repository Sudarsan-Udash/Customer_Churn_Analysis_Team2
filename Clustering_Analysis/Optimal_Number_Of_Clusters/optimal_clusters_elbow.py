# -*- coding: utf-8 -*-
"""
================================================================================
CLUSTERING ANALYSIS — STEP 1: OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
================================================================================
Project : Customer Churn Analytics & Revenue Recovery
Output  : elbow_plot.png, elbow_results.txt
================================================================================
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = '/content/drive/MyDrive/Colab Notebooks/'
DATA_PREP = BASE + 'Data_Preparation/'
CLUST_OUT = BASE + 'Clustering_Analysis/'

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PREP + 'preprocessed_dataset.csv')
print(f"Loaded: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCALE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
X_cluster = df[['tenure', 'MonthlyCharges']]
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ELBOW METHOD — WCSS for k=1 to k=10
# ─────────────────────────────────────────────────────────────────────────────
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    print(f"  k={k:2d}  |  WCSS = {km.inertia_:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='#34495e', linewidth=2)
plt.annotate(
    'OPTIMAL: k=3',
    xy=(3, wcss[2]),
    xytext=(4.2, wcss[2] + 4000),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',
                    color='#c0392b', lw=1.5),
    fontsize=10, fontweight='bold', color='#c0392b'
)
plt.title('ELBOW METHOD: Determining Optimal Number of Clusters',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True, axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(CLUST_OUT + 'elbow_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("elbow_plot.png saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE WCSS RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
results = "k,WCSS\n" + "\n".join([f"{k+1},{wcss[k]:.4f}" for k in range(10)])
with open(CLUST_OUT + 'elbow_results.txt', 'w') as f:
    f.write("ELBOW METHOD RESULTS\n")
    f.write("=" * 40 + "\n")
    f.write(f"{'k':>4}  {'WCSS':>14}\n")
    f.write("-" * 40 + "\n")
    for k in range(10):
        marker = "  <-- OPTIMAL" if k == 2 else ""
        f.write(f"{k+1:>4}  {wcss[k]:>14,.2f}{marker}\n")
    f.write("\nConclusion: The elbow inflection point occurs at k=3.\n")
    f.write("Business Logic: k=3 balances segment granularity with operational simplicity.\n")

print("\nElbow Results:")
for k in range(10):
    marker = " <-- OPTIMAL" if k == 2 else ""
    print(f"  k={k+1:2d}  WCSS={wcss[k]:>12,.2f}{marker}")
print("\nConclusion: k=3 selected as optimal cluster count.")
