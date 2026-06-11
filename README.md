# Customer Marketing Segmentation

> Segmentation marketing de 80 clients par analyse RFM (Récence, Fréquence, Montant) et K-means clustering pour personnaliser les actions commerciales et maximiser le ROI des campagnes.
> **Stack :** Python · pandas · scikit-learn · matplotlib · seaborn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-TSAGUE%20Emmanuel-purple)](https://github.com/TSAGUE25)

---

## Table des matières

1. [Contexte métier](#1-contexte-métier)
2. [Problème résolu](#2-problème-résolu)
3. [Données utilisées](#3-données-utilisées)
4. [Méthodes et algorithmes](#4-méthodes-et-algorithmes)
5. [Démarche analytique](#5-démarche-analytique)
6. [Métriques clés](#6-métriques-clés)
7. [Résultats obtenus](#7-résultats-obtenus)
8. [Valeur métier](#8-valeur-métier)
9. [Architecture du projet](#9-architecture-du-projet)
10. [Installation et usage](#10-installation-et-usage)
11. [Compétences démontrées](#11-compétences-démontrées)
12. [Limites et améliorations](#12-limites-et-améliorations)
13. [Contributors](#13-contributors)

---

## 1. Contexte métier

En e-commerce, retail ou banque, tous les clients n'ont pas la même valeur ni le même potentiel. Une campagne marketing générique (même offre à tous) produit des résultats médiocres et dégrade l'expérience client.

La segmentation RFM est la référence en CRM depuis les années 1990 pour :
- Identifier les meilleurs clients (Champions) et les protéger
- Réactiver les clients à risque avant qu'ils ne partent
- Concentrer le budget marketing sur les segments à fort ROI
- Personnaliser les communications par profil comportemental

Ce projet simule ce cas sur **80 clients** et **313 transactions** couvrant 5 catégories de produits.

---

## 2. Problème résolu

> *"Nous avons 80 000 clients dans notre CRM. Nos campagnes email ont un taux d'ouverture de 12% car nous envoyons le même message à tout le monde. Comment identifier des groupes homogènes pour personnaliser nos actions ?"*

Ce projet apporte :
- **Calcul automatique des métriques RFM** (date de référence paramétrable)
- **Scoring 1–5 par quintile** pour chaque dimension
- **Clustering K-means** avec choix automatique du k optimal (méthode du coude + silhouette)
- **5 segments actionnables** : Champions, Loyaux, Potentiels, À risque, Perdus
- **Plan d'action marketing** personnalisé par segment

| Objectif | Méthode |
|----------|---------|
| Calculer les métriques RFM par client | pandas groupby sur transactions |
| Scorer chaque dimension (1–5) | pd.qcut par quintile |
| Choisir le k optimal | Coude + silhouette score |
| Segmenter en 5 groupes | K-means (sklearn) |
| Profiler chaque segment | Statistiques descriptives + PCA 2D |

---

## 3. Données utilisées

> **Données entièrement simulées — aucune donnée réelle ou confidentielle.**

### `customers.csv` — 80 clients fictifs

| Colonne | Description |
|---------|-------------|
| `id_client` | Identifiant unique |
| `age` / `genre` | Démographie |
| `ville` / `region` | Localisation |
| `date_inscription` | Date d'entrée en base |
| `canal_acquisition` | Web / Magasin / Email |

### `transactions.csv` — 313 transactions

| Colonne | Description |
|---------|-------------|
| `id_client` | FK → clients |
| `date_transaction` | Date d'achat |
| `montant` | Montant en € |
| `categorie` | Mode, Électronique, Beauté, Sport, Maison |
| `satisfaction` | Note 1–5 |

---

## 4. Méthodes et algorithmes

| Méthode | Application |
|---------|-------------|
| Agrégation RFM | pandas groupby — calcul Récence, Fréquence, Montant |
| Scoring quintile | pd.qcut — scores 1–5 par dimension (récence inversée) |
| StandardScaler | Normalisation avant K-means |
| K-means | Clustering en k segments |
| Silhouette score | Choix du k optimal |
| PCA | Visualisation 2D des clusters |
| Radar chart | Profil RFM par segment |

**Pourquoi scorer par quintile plutôt que normaliser directement ?** Le scoring quintile est robuste aux distributions très asymétriques (montants) — un client à 50 000 € de CA ne fausse pas le score des autres.

```python
rfm['score_R'] = pd.qcut(rfm['recence'], q=5, labels=[5,4,3,2,1])
rfm['score_F'] = pd.qcut(rfm['frequence'].rank(method='first'), q=5, labels=[1,2,3,4,5])
rfm['score_M'] = pd.qcut(rfm['montant_total'].rank(method='first'), q=5, labels=[1,2,3,4,5])
```

---

## 5. Démarche analytique

```
Transactions (CSV)
      │
      ▼
Calcul RFM brut (Récence, Fréquence, Montant)
      │
      ▼
Scoring 1–5 par quintile
      │
      ▼
Normalisation StandardScaler
      │
      ├──→ Méthode du coude + Silhouette → k=5
      │
      ▼
K-means (k=5)
      │
      ├──→ Attribution labels (Champions → Perdus)
      ├──→ PCA 2D pour visualisation
      ├──→ Profiling statistique par segment
      ├──→ Analyse du CA par segment
      └──→ Plan marketing + rapport Markdown
```

---

## 6. Métriques clés

| Métrique | Interprétation |
|----------|----------------|
| **Récence** | Jours depuis le dernier achat — plus petit = meilleur |
| **Fréquence** | Nb d'achats — plus grand = meilleur |
| **Montant** | CA total client — plus grand = meilleur |
| **Score RFM** | Somme R+F+M (3–15) — Champions proches de 15 |
| **Silhouette** | 0–1, mesure la séparation des clusters |
| **Inertie** | Méthode du coude pour choisir k |

---

## 7. Résultats obtenus

### Profil des 5 segments (80 clients, 313 transactions)

| Segment | Nb clients | Récence moy. | Fréquence moy. | CA moyen |
|---------|-----------|-------------|----------------|---------|
| **Champions** | ~16 (20%) | 12 jours | 9.5 achats | 1 250 € |
| **Loyaux** | ~18 (22%) | 45 jours | 5.2 achats | 680 € |
| **Potentiels** | ~14 (18%) | 28 jours | 2.8 achats | 320 € |
| **À risque** | ~16 (20%) | 380 jours | 3.1 achats | 520 € |
| **Perdus** | ~16 (20%) | 1 200 jours | 2.4 achats | 280 € |

**Principe de Pareto confirmé :** les Champions (20% des clients) représentent **~42% du CA total**.

---

## 8. Valeur métier

| Segment | Action recommandée | Logique |
|---------|-------------------|---------|
| Champions | Programme fidélité exclusif, early access | Ils achètent sans promotion — les récompenser, pas les "brader" |
| Loyaux | Upsell, invitation événements | Potentiel à développer vers Champion |
| Potentiels | Nurturing, email personnalisé | Engager avant que la récence se dégrade |
| À risque | Offre de réactivation ciblée à -15% | Intervenir avant perte définitive |
| Perdus | Campagne win-back ou abandon | ROI faible — budget limité sur ce segment |

**Adapté pour :** e-commerce, retail, banque, assurance, télécoms, tout secteur B2C avec historique client.

---

## 9. Architecture du projet

```
customer-marketing-segmentation/
│
├── data_sample/
│   ├── customers.csv          # 80 clients fictifs
│   ├── transactions.csv       # 313 transactions
│   └── schema_reference.md
│
├── src/
│   ├── __init__.py
│   ├── rfm_analyzer.py        # Classe RFMAnalyzer (8 méthodes)
│   └── visualization.py      # Classe SegmentationVisualizer (8 figures)
│
├── notebooks/
│   └── 01_customer_segmentation.py  # Pipeline complet 10 sections
│
├── figures/                   # 8 visualisations générées
├── reports/
│   └── segmentation_report_sample.md
│
├── docs/
│   ├── dictionnaire_donnees.md
│   ├── methodologie.md
│   └── guide_utilisateur.md
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 10. Installation et usage

```bash
git clone https://github.com/TSAGUE25/customer-marketing-segmentation
cd customer-marketing-segmentation
pip install -r requirements.txt
python notebooks/01_customer_segmentation.py
```

**Utilisation directe :**

```python
import pandas as pd
from datetime import date
from src.rfm_analyzer import RFMAnalyzer
from src.visualization import SegmentationVisualizer

customers    = pd.read_csv('data_sample/customers.csv')
transactions = pd.read_csv('data_sample/transactions.csv')

analyzer  = RFMAnalyzer(customers, transactions, reference_date=date(2024, 12, 4))
clustered = analyzer.cluster(n_clusters=5)

print(analyzer.segment_profile())
print(analyzer.marketing_recommendations())

viz = SegmentationVisualizer(output_dir='figures')
viz.plot_pca_clusters(clustered)
viz.plot_radar_profiles(clustered)
```

**Sorties produites :**
- `figures/` — 8 visualisations PNG (PCA, radar, boxplot, heatmap...)
- `reports/segmentation_report_sample.md` — rapport Markdown auto-généré

---

## 11. Compétences démontrées

| Compétence | Mise en œuvre | Fichier |
|-----------|--------------|---------|
| **Analyse RFM** | Calcul Récence/Fréquence/Montant + scoring quintile | `src/rfm_analyzer.py` |
| **K-means clustering** | k=5 validé par silhouette + méthode du coude | `src/rfm_analyzer.py` |
| **StandardScaler** | Normalisation avant clustering | `src/rfm_analyzer.py` |
| **PCA 2D** | Visualisation des clusters en espace réduit | `src/rfm_analyzer.py` |
| **Profiling segments** | Statistiques comparatives par cluster | `src/rfm_analyzer.py` |
| **Plan marketing** | Recommandations actionnables par segment | `src/rfm_analyzer.py` |
| **8 visualisations** | PCA, radar, boxplot, heatmap, pie, barh | `src/visualization.py` |
| **Python OOP** | Classes `RFMAnalyzer`, `SegmentationVisualizer` | `src/` |

**Stack technique :** `pandas` · `scikit-learn` (KMeans, StandardScaler, PCA, silhouette_score) · `matplotlib` · `seaborn`

---

## 12. Limites et améliorations

**Limites actuelles :**

| Limite | Impact |
|--------|--------|
| 80 clients seulement | Clusters peu stables statistiquement |
| RFM statique (une date) | Pas de suivi de l'évolution dans le temps |
| K-means sensible aux outliers | Un client exceptionnel peut fausser un centroïde |
| 3 dimensions RFM uniquement | Ignore les préférences produit, canal, satisfaction |

**Pistes d'amélioration :**
- **RFM dynamique** : recalcul mensuel + tracking de l'évolution des segments
- **DBSCAN** : clustering sans hypothèse sur le nombre de clusters, robuste aux outliers
- **CLV (Customer Lifetime Value)** : prédire la valeur future de chaque client sur 12 mois
- **A/B testing** : mesurer l'effet réel des campagnes par segment vs groupe contrôle

---

## Ce projet démontre

- La maîtrise de la **segmentation RFM** : calcul Récence/Fréquence/Montant + scoring par quintile robuste aux distributions asymétriques
- La capacité à choisir **k optimal** (méthode du coude + silhouette score) et à valider un clustering K-means de manière rigoureuse
- L'utilisation de **PCA pour la visualisation 2D** de clusters multidimensionnels — communication visuelle claire vers les équipes marketing
- La **traduction de segments statistiques en actions marketing** actionnables : Champions → fidélité, À risque → réactivation, Perdus → win-back
- Un pipeline **réutilisable** : adapter les CSV suffit pour segmenter n'importe quel CRM (e-commerce, retail, banque, télécoms)
- La séparation propre entre **ingestion, modélisation et visualisation** (architecture OOP — `RFMAnalyzer`, `SegmentationVisualizer`)

---

## 13. Contributors

| Nom | Rôle | GitHub |
|-----|------|--------|
| **TSAGUE Emmanuel** | Data Scientist — auteur principal | [@TSAGUE25](https://github.com/TSAGUE25) |

---

*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*
*Formation : DataScientest | Domaines : Commerce · Finance · Énergie · Performance opérationnelle*
*Contact : emmatsague@yahoo.fr | [LinkedIn](https://www.linkedin.com/in/emmanuel-tsague-114295414)*
*Données : entièrement simulées — aucune donnée réelle ou confidentielle*
