# -*- coding: utf-8 -*-
"""
================================================================================
CLUSTERING ANALYSIS — STEP 3: CLUSTER VISUALISATIONS & LABELLING
================================================================================
Project : Customer Churn Analytics & Revenue Recovery
Outputs : 5 PNG visualisations + cluster_insights.txt
================================================================================
PLOTS PRODUCED:
  1. cluster_behavioural_map.png      — Persona scatter + centroids
  2. cluster_simple.png               — Basic cluster scatter (numeric IDs)
  3. onboarding_cliff.png             — 10-Month tenure histogram
  4. revenue_leak_by_persona.png      — Revenue leakage per persona
  5. senior_fiber_trap.png            — Senior-Fiber micro-segmentation
================================================================================
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

BASE       = '/content/drive/MyDrive/Colab Notebooks/'
DATA_PREP  = BASE + 'Data_Preparation/'
CLUST_OUT  = BASE + 'Clustering_Analysis/'

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA AND MODEL
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CLUST_OUT + 'clustered_dataset.csv')

with open(CLUST_OUT + 'kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open(CLUST_OUT + 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"Loaded dataset: {df.shape}")
print(f"Personas: {df['Persona'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: BEHAVIOURAL MAP — SCATTER WITH PERSONA LABELS & CENTROIDS
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
colors  = {'Loyal High-Rollers': '#f39c12', 'At-Risk Newbies': '#e74c3c', 'Stable Budgeteers': '#3498db'}
for persona, color in colors.items():
    subset = df[df['Persona'] == persona]
    ax.scatter(subset['tenure'], subset['MonthlyCharges'],
               c=color, label=persona, alpha=0.4, s=40, edgecolors='none')

centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1],
           c='black', s=250, marker='X', zorder=5, label='Segment Centre')

# Annotate centroids with persona names
persona_labels = {0: "Loyal\nHigh-Rollers", 1: "At-Risk\nNewbies", 2: "Stable\nBudgeteers"}
for i, (cx, cy) in enumerate(centers):
    ax.annotate(persona_labels[i], (cx, cy),
                textcoords="offset points", xytext=(10, 8),
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

ax.set_title("CUSTOMER SEGMENTATION: Behavioural Map by Persona",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Customer Tenure (Months)")
ax.set_ylabel("Monthly Charges ($)")
ax.legend(title="Customer Personas", loc='upper left', frameon=True)
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(CLUST_OUT + 'cluster_behavioural_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("cluster_behavioural_map.png saved.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: SIMPLE CLUSTER SCATTER (numeric cluster IDs)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['tenure'], y=df['MonthlyCharges'],
                hue=df['Cluster'], palette='Set2', alpha=0.6)
plt.title("Customer Segmentation using K-Means Clustering (k=3)")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.savefig(CLUST_OUT + 'cluster_simple.png', dpi=150, bbox_inches='tight')
plt.show()
print("cluster_simple.png saved.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: THE 10-MONTH ONBOARDING CLIFF
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='tenure', hue='Churn',
             multiple='stack', palette='coolwarm', bins=30)
plt.axvline(9.8, color='red', linestyle='--', linewidth=3, label='9.8-Month Risk Average')
plt.legend(title='Customer Status', labels=['Left (Churn)', 'Stayed', '9.8-Month Risk Average'])
plt.title("STRATEGIC INSIGHT: The 10-Month 'Onboarding Cliff'",
          fontsize=14, fontweight='bold')
plt.xlabel("Months with Company (Tenure)")
plt.ylabel("Customer Volume (Count)")
plt.tight_layout()
plt.savefig(CLUST_OUT + 'onboarding_cliff.png', dpi=150, bbox_inches='tight')
plt.show()
print("onboarding_cliff.png saved.")
print("CRUX: Churn is highest BEFORE the 10-month red line.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: REVENUE LEAK BY PERSONA (annotated bars)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))
ax = sns.barplot(data=df, x='Cluster_Name', y='CustomerValue',
                 hue='Churn', palette='viridis', estimator=sum, errorbar=None)
plt.legend(title='Financial Status',
           labels=['Active Revenue (Protected)', 'Lost Revenue (Churned)'])
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'${p.get_height():,.0f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9),
                    textcoords='offset points', fontsize=9, fontweight='bold')
plt.title("STRATEGIC INSIGHT: Total Revenue Volume vs. Churn Status",
          fontsize=14, fontweight='bold')
plt.ylabel("Cumulative Lifetime Revenue ($)")
plt.xlabel("Customer Persona")
plt.tight_layout()
plt.savefig(CLUST_OUT + 'revenue_leak_by_persona.png', dpi=150, bbox_inches='tight')
plt.show()
print("revenue_leak_by_persona.png saved.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: SENIOR-FIBER MICRO-SEGMENTATION TRAP
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
risk_data = df.groupby('Senior_Fiber_Risk')['Churn'].mean() * 100
ax = sns.barplot(x=risk_data.index, y=risk_data.values,
                 palette=['#34495e', '#e74c3c'])
plt.xticks([0, 1], ['General Population', 'SENIOR-FIBER RISK GROUP'])
plt.ylabel("Churn Rate (%)", fontweight='bold')
plt.title("MICRO-SEGMENTATION: The 'Senior-Fiber' Churn Trap",
          fontsize=14, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9),
                textcoords='offset points', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CLUST_OUT + 'senior_fiber_trap.png', dpi=150, bbox_inches='tight')
plt.show()
print("senior_fiber_trap.png saved.")

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER INSIGHTS TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
total_loss  = df[df['Churn_numeric'] == 1]['CustomerValue'].sum()
newbie_loss = df[(df['Cluster_Name'] == 'The At-Risk Newbies') & (df['Churn'] == 1)]['CustomerValue'].sum()
risk_rev    = df[df['Senior_Fiber_Risk'] == 1]['CustomerValue'].sum()

summary = df.groupby('Persona').agg({
    'tenure': 'mean', 'MonthlyCharges': 'mean', 'Churn_numeric': 'mean'
})

insights = f"""
================================================================================
CLUSTER INSIGHTS & INTERPRETATION
================================================================================
Author          : Saurav Kumar
Project         : Customer Churn Analytics & Revenue Recovery
Model           : K-Means (k=3, k-means++, random_state=42)
Features        : tenure (scaled), MonthlyCharges (scaled)
--------------------------------------------------------------------------------

PERSONA 1 — LOYAL HIGH-ROLLERS (Cluster 0)
  Avg Tenure     : {summary.loc['Loyal High-Rollers','tenure']:.1f} months
  Avg Bill       : ${summary.loc['Loyal High-Rollers','MonthlyCharges']:.2f}/month
  Churn Rate     : {summary.loc['Loyal High-Rollers','Churn_numeric']*100:.1f}%
  Interpretation : Long-term, high-paying VIPs. Lowest churn risk. High-touch
                   support and loyalty rewards recommended to maintain retention.

PERSONA 2 — AT-RISK NEWBIES (Cluster 1)
  Avg Tenure     : {summary.loc['At-Risk Newbies','tenure']:.1f} months
  Avg Bill       : ${summary.loc['At-Risk Newbies','MonthlyCharges']:.2f}/month
  Churn Rate     : {summary.loc['At-Risk Newbies','Churn_numeric']*100:.1f}%
  Revenue at Risk: ${newbie_loss:,.2f}
  Interpretation : New customers with high bills and no loyalty yet established.
                   HIGHEST PRIORITY. Intervene at the 15-month mark with
                   discounted bundle offers before the 17.5-month cliff.

PERSONA 3 — STABLE BUDGETEERS (Cluster 2)
  Avg Tenure     : {summary.loc['Stable Budgeteers','tenure']:.1f} months
  Avg Bill       : ${summary.loc['Stable Budgeteers','MonthlyCharges']:.2f}/month
  Churn Rate     : {summary.loc['Stable Budgeteers','Churn_numeric']*100:.1f}%
  Interpretation : Long-tenured but price-sensitive. Stable and low-risk.
                   Maintain with low-cost loyalty perks. Avoid price increases.

COMPANY-WIDE SUMMARY
  Total Revenue Leakage    : ${total_loss:,.2f}
  Senior-Fiber Revenue Risk: ${risk_rev:,.2f}
  Critical Danger Zone     : Fiber Optic + Month-to-Month (40%+ churn)
  Recommended Action       : Intervene at 15-month mark for Cluster 1 customers.
                             Converting them to 1-Year bundles could recover
                             up to 60% of the $2.86M revenue at risk.

VISUALISATIONS SAVED
  cluster_behavioural_map.png   — Persona scatter + centroids
  cluster_simple.png            — Basic scatter with numeric cluster IDs
  onboarding_cliff.png          — 10-Month Onboarding Cliff histogram
  revenue_leak_by_persona.png   — Revenue leakage per persona ($)
  senior_fiber_trap.png         — Senior-Fiber micro-segmentation bar chart
================================================================================
"""
print(insights)
with open(CLUST_OUT + 'cluster_insights.txt', 'w') as f:
    f.write(insights)
print("cluster_insights.txt saved.")
