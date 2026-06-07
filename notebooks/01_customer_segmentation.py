"""
Segmentation Marketing Clients — RFM + K-means
Cas 10 — Portfolio Data Science | Emmanuel TSAGUE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.rfm_analyzer import RFMAnalyzer
from src.visualization import SegmentationVisualizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data_sample')
FIG_DIR   = os.path.join(BASE_DIR, 'figures')
REP_DIR   = os.path.join(BASE_DIR, 'reports')
os.makedirs(REP_DIR, exist_ok=True)

# ======================================================================
# 0. CHARGEMENT
# ======================================================================
customers     = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
transactions  = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'))

print(f"Clients     : {customers.shape}")
print(f"Transactions: {transactions.shape}")
print(f"Période     : {transactions['date_transaction'].min()} → {transactions['date_transaction'].max()}")

# ======================================================================
# 1. EXPLORATION
# ======================================================================
print("\n" + "="*60)
print("1. EXPLORATION")
print("="*60)
print(f"\nCA total simulé  : {transactions['montant'].sum():,.0f} €")
print(f"Panier moyen     : {transactions['montant'].mean():.2f} €")
print(f"Transactions/client : {transactions.groupby('id_client')['id_transaction'].count().mean():.1f}")
print("\nRépartition par catégorie :")
print(transactions.groupby('categorie')['montant'].agg(['count','sum']).rename(
    columns={'count':'nb','sum':'ca'}).sort_values('ca', ascending=False))
print("\nRépartition par canal :")
print(transactions.groupby('canal')['montant'].agg(['count','sum']))

# ======================================================================
# 2. CALCUL RFM
# ======================================================================
print("\n" + "="*60)
print("2. CALCUL RFM (date ref = 2024-12-04)")
print("="*60)

from datetime import date
analyzer = RFMAnalyzer(customers, transactions, reference_date=date(2024, 12, 4))
rfm = analyzer.compute_rfm()

print(f"\nStatistiques RFM :")
print(rfm[['recence','frequence','montant_total']].describe().round(1))

# ======================================================================
# 3. SCORES RFM
# ======================================================================
print("\n" + "="*60)
print("3. SCORES RFM (1-5)")
print("="*60)
rfm = analyzer.compute_rfm_scores()
print(f"\nDistribution score total :")
print(rfm['score_rfm'].value_counts().sort_index())

# ======================================================================
# 4. CHOIX DU K OPTIMAL
# ======================================================================
print("\n" + "="*60)
print("4. CHOIX DU K OPTIMAL")
print("="*60)
k_range, inertias, silhouettes = analyzer.find_optimal_k(k_range=range(2, 8))
best_k = k_range[int(np.argmax(silhouettes))]
print(f"\nScore silhouette par k :")
for k, s in zip(k_range, silhouettes):
    marker = " ← OPTIMAL" if k == best_k else ""
    print(f"  k={k} : silhouette={s:.3f}{marker}")

# ======================================================================
# 5. CLUSTERING K-MEANS (k=5)
# ======================================================================
print("\n" + "="*60)
print("5. CLUSTERING K-MEANS (k=5 segments)")
print("="*60)
clustered = analyzer.cluster(n_clusters=5)
print(f"\nDistribution des segments :")
print(clustered['segment'].value_counts())

# ======================================================================
# 6. PROFIL DES SEGMENTS
# ======================================================================
print("\n" + "="*60)
print("6. PROFIL DES SEGMENTS")
print("="*60)
profile = analyzer.segment_profile()
print(f"\n{profile.to_string()}")

revenue = analyzer.segment_revenue()
print(f"\nCA par segment :")
print(revenue.to_string())

# ======================================================================
# 7. RECOMMANDATIONS MARKETING
# ======================================================================
print("\n" + "="*60)
print("7. RECOMMANDATIONS MARKETING")
print("="*60)
recs = analyzer.marketing_recommendations()
print(f"\n{recs.to_string(index=False)}")

# ======================================================================
# 8. SYNTHESE
# ======================================================================
print("\n" + "="*60)
print("8. SYNTHESE DU PARC CLIENT")
print("="*60)
s = analyzer.summary()
print(f"\nClients analysés    : {s['nb_clients_total']}")
print(f"CA total            : {s['ca_total']:,.0f} €")
print(f"CA moyen/client     : {s['ca_moyen_client']:,.0f} €")
print(f"Récence moyenne     : {s['recence_moyenne_jours']:.0f} jours")
print(f"Fréquence moyenne   : {s['frequence_moyenne']:.1f} transactions")
print(f"Champions           : {s['champions_pct']}% des clients")
print(f"Clients perdus      : {s['perdus_pct']}% des clients")

# ======================================================================
# 9. VISUALISATIONS
# ======================================================================
print("\n" + "="*60)
print("9. GENERATION DES FIGURES")
print("="*60)
viz = SegmentationVisualizer(output_dir=FIG_DIR)
viz.plot_elbow(k_range, inertias, silhouettes)
viz.plot_pca_clusters(clustered)
viz.plot_rfm_boxplots(clustered)
viz.plot_segment_distribution(profile, revenue)
viz.plot_rfm_heatmap(clustered)
viz.plot_purchase_timeline(transactions, clustered)
viz.plot_radar_profiles(clustered)
viz.plot_marketing_plan(recs)
print("\n8 figures générées dans figures/")

# ======================================================================
# 10. RAPPORT MARKDOWN
# ======================================================================
print("\n" + "="*60)
print("10. RAPPORT MARKDOWN")
print("="*60)

lines = [
    "# Rapport de Segmentation Marketing Clients",
    "",
    f"**Date :** 2024-12-04  ",
    f"**Périmètre :** {s['nb_clients_total']} clients | {len(transactions)} transactions  ",
    f"**CA analysé :** {s['ca_total']:,.0f} €",
    "",
    "> *Données simulées à des fins pédagogiques — aucune donnée réelle.*",
    "",
    "---",
    "",
    "## 1. Synthèse exécutive",
    "",
    "| Indicateur | Valeur |",
    "|-----------|--------|",
    f"| Clients analysés | **{s['nb_clients_total']}** |",
    f"| Segments identifiés | **{s['nb_segments']}** |",
    f"| CA total | **{s['ca_total']:,.0f} €** |",
    f"| CA moyen par client | **{s['ca_moyen_client']:,.0f} €** |",
    f"| Récence moyenne | **{s['recence_moyenne_jours']:.0f} jours** |",
    f"| Fréquence moyenne | **{s['frequence_moyenne']:.1f} transactions** |",
    f"| Champions | **{s['champions_pct']}%** du portefeuille |",
    f"| Clients à risque / perdus | **{100 - s['champions_pct'] - 20:.0f}%** |",
    "",
    "---",
    "",
    "## 2. Profil des 5 segments",
    "",
    "| Segment | Nb clients | % | Récence (j) | Fréquence | CA moyen (€) | Score RFM |",
    "|---------|-----------|---|------------|-----------|-------------|-----------|",
]

from src.rfm_analyzer import SEGMENT_LABELS, SEGMENT_ORDER
for seg in SEGMENT_ORDER:
    if seg in profile.index:
        row = profile.loc[seg]
        lines.append(
            f"| **{seg}** | {int(row['nb_clients'])} | {row['pct_clients']}% | "
            f"{row['recence_moy']:.0f} | {row['frequence_moy']:.1f} | "
            f"{row['montant_total_moy']:,.0f} | {row['score_rfm_moy']:.1f} |"
        )

lines += [
    "",
    "---",
    "",
    "## 3. Plan d'action marketing",
    "",
    "| Segment | Description | Action recommandée |",
    "|---------|-------------|-------------------|",
]
for _, row in recs.iterrows():
    lines.append(f"| **{row['segment']}** | {row['description']} | {row['action_recommandee']} |")

lines += [
    "",
    "---",
    "",
    "## 4. Distribution du CA par segment",
    "",
    "| Segment | CA total (€) | % CA |",
    "|---------|-------------|------|",
]
for seg, row in revenue.iterrows():
    lines.append(f"| **{seg}** | {row['ca_total']:,.0f} | {row['pct_ca']}% |")

lines += [
    "",
    "---",
    "",
    "## 5. Recommandations stratégiques",
    "",
    "1. **Champions** : programme ambassadeur, accès early access, parrainage",
    "2. **Loyaux** : fidélisation active, montée en gamme, offres croisées",
    "3. **Potentiels** : séquence d'onboarding email, réduction 2e achat",
    "4. **À risque** : campagne win-back, offre personnalisée, enquête satisfaction",
    "5. **Perdus** : tentative réactivation unique, sinon désabonnement propre",
    "",
    "---",
    "",
    "*Rapport généré par Customer Marketing Segmentation Framework*  ",
    "*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*",
]

report_path = os.path.join(REP_DIR, 'segmentation_report_sample.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Rapport : {report_path}")
print("\n=== ANALYSE TERMINÉE ===")
