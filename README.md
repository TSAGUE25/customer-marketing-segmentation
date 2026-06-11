# Customer Marketing Segmentation

> Segmentation marketing de 80 clients par analyse RFM (Récence, Fréquence, Montant) et K-means clustering pour personnaliser les actions commerciales.  
> **Stack :** Python · pandas · scikit-learn · matplotlib · seaborn

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-Data%20Science-orange)](https://github.com/TSAGUE25)

---

## Table des matières

1. [Titre et accroche](#1-titre-et-accroche)
2. [Contexte métier](#2-contexte-métier)
3. [Pourquoi ce projet existe](#3-pourquoi-ce-projet-existe)
4. [Problème métier](#4-problème-métier)
5. [Objectifs](#5-objectifs)
6. [Données utilisées](#6-données-utilisées)
7. [Préparation des données](#7-préparation-des-données)
8. [Méthodes et algorithmes](#8-méthodes-et-algorithmes)
9. [Démarche analytique](#9-démarche-analytique)
10. [Métriques clés](#10-métriques-clés)
11. [Explication des métriques](#11-explication-des-métriques)
12. [Résultats obtenus](#12-résultats-obtenus)
13. [Valeur métier](#13-valeur-métier)
14. [Limites du projet](#14-limites-du-projet)
15. [Améliorations possibles](#15-améliorations-possibles)
16. [Architecture du dépôt](#16-architecture-du-dépôt)
17. [README technique](#17-readme-technique)
18. [Version CV](#18-version-cv)
19. [Version entretien](#19-version-entretien)
20. [Version portfolio](#20-version-portfolio)
21. [Post LinkedIn](#21-post-linkedin)
22. [Questions d'entretien](#22-questions-dentretien)
23. [Compétences démontrées](#23-compétences-démontrées)
24. [Tableau compétences / preuves](#24-tableau-compétences--preuves)
25. [Conseils GitHub](#25-conseils-github)

---

## 1. Titre et accroche

**Customer Marketing Segmentation** — Pipeline Python complet de segmentation client par analyse RFM et K-means : calcul des scores Récence/Fréquence/Montant, détermination du k optimal (méthode du coude + silhouette), clustering en 5 segments actionnables et plan d'action marketing personnalisé par segment.

> *Ce projet démontre la capacité à transformer un historique transactionnel brut en recommandations marketing concrètes à fort ROI.*

---

## 2. Contexte métier

En e-commerce, retail ou banque, tous les clients n'ont pas la même valeur ni le même potentiel. Une campagne marketing générique (même offre à tous) produit des résultats médiocres et dégrade l'expérience client.

La segmentation RFM est une technique éprouvée depuis les années 1990 (Bain & Company) qui reste la référence en CRM pour :
- Identifier les meilleurs clients (Champions) et les protéger
- Réactiver les clients à risque avant qu'ils ne partent
- Concentrer le budget marketing sur les segments à fort ROI
- Personnaliser les communications par profil comportemental

Ce projet simule ce cas sur **80 clients** et **313 transactions** couvrant 5 catégories (Mode, Électronique, Beauté, Sport, Maison, Alimentaire).

---

## 3. Pourquoi ce projet existe

**Problème concret :** Un responsable marketing reçoit une liste de clients et leur historique d'achat. Sans segmentation, il envoie la même newsletter promotionnelle à tous — un Champion qui achète tous les mois reçoit la même offre qu'un client inactif depuis 2 ans.

**Ce que ce projet apporte :**
- Calcul automatique des métriques RFM (date de référence paramétrable)
- Scoring 1–5 par quintile pour chaque dimension
- Clustering K-means avec choix automatique du k optimal
- 5 segments actionnables avec nom, description et action marketing associée
- 8 visualisations professionnelles + rapport Markdown automatisé

---

## 4. Problème métier

> *"Nous avons 80 000 clients dans notre CRM. Nos campagnes email ont un taux d'ouverture de 12% et un ROI médiocre car nous envoyons le même message à tout le monde. Comment identifier des groupes homogènes pour personnaliser nos actions ?"*

**Traduction analytique :**
- Calculer pour chaque client : Récence (dernière visite), Fréquence (nb achats), Montant (CA total)
- Normaliser et scorer ces 3 dimensions
- Appliquer K-means pour former des clusters homogènes
- Profiler chaque cluster et lui associer une stratégie marketing

---

## 5. Objectifs

| # | Objectif | Méthode |
|---|----------|---------|
| 1 | Calculer les métriques RFM par client | pandas groupby sur transactions |
| 2 | Scorer chaque dimension (1–5) | pd.qcut par quintile |
| 3 | Choisir le k optimal | Méthode du coude + score silhouette |
| 4 | Segmenter en 5 groupes | K-means (sklearn) |
| 5 | Visualiser les segments en 2D | PCA normalisé |
| 6 | Profiler chaque segment | Statistiques descriptives |
| 7 | Quantifier la valeur par segment | CA total et moyen |
| 8 | Produire un plan marketing | Tableau de recommandations |

---

## 6. Données utilisées

> **Données entièrement simulées — aucune donnée réelle ou confidentielle.**

### 6.1 customers.csv — 80 clients fictifs

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `id_client` | Identifiant unique | C001 |
| `age` | Âge du client | 34 |
| `genre` | F / M | F |
| `ville` / `region` | Localisation | Paris, Ile-de-France |
| `date_inscription` | Date d'entrée en base | 2020-03-15 |
| `canal_acquisition` | Web / Magasin / Email | Web |
| `categorie_principale` | Catégorie de prédilection | Mode |

### 6.2 transactions.csv — 313 transactions

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `id_transaction` | Identifiant unique | T0001 |
| `id_client` | FK → clients | C001 |
| `date_transaction` | Date d'achat | 2024-11-28 |
| `montant` | Montant en € | 89.90 |
| `categorie` | Catégorie achetée | Mode |
| `canal` | Web / Magasin / Email | Web |
| `satisfaction` | Note 1–5 | 5 |

---

## 7. Préparation des données

### Calcul RFM (date référence : 2024-12-04)

```python
rfm = (
    transactions
    .groupby('id_client')
    .agg(
        derniere_transaction=('date_transaction', 'max'),
        frequence=('id_transaction', 'count'),
        montant_total=('montant', 'sum'),
    )
)
rfm['recence'] = (pd.Timestamp('2024-12-04') - rfm['derniere_transaction']).dt.days
```

### Scoring par quintile

```python
# Récence inversée : plus faible = meilleur → score 5
rfm['score_R'] = pd.qcut(rfm['recence'], q=5, labels=[5,4,3,2,1])
rfm['score_F'] = pd.qcut(rfm['frequence'].rank(method='first'), q=5, labels=[1,2,3,4,5])
rfm['score_M'] = pd.qcut(rfm['montant_total'].rank(method='first'), q=5, labels=[1,2,3,4,5])
```

### Normalisation avant clustering

```python
scaler = StandardScaler()
X = scaler.fit_transform(rfm[['recence', 'frequence', 'montant_total']])
```

---

## 8. Méthodes et algorithmes

| Méthode | Bibliothèque | Application |
|---------|-------------|-------------|
| Agrégation RFM | pandas | Calcul des 3 métriques |
| Scoring quintile | pandas.qcut | Scores 1–5 par dimension |
| StandardScaler | scikit-learn | Normalisation avant K-means |
| K-means | scikit-learn | Clustering en k segments |
| Silhouette score | scikit-learn | Choix du k optimal |
| PCA | scikit-learn | Visualisation 2D des clusters |
| Radar chart | matplotlib | Profil RFM par segment |
| Boxplot / Heatmap | seaborn | Distribution RFM |

---

## 9. Démarche analytique

```
Transactions (CSV)
      │
      ▼
Calcul RFM brut
(Récence, Fréquence, Montant)
      │
      ▼
Scoring 1-5 par quintile
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
      │
      ├──→ PCA 2D pour visualisation
      │
      ├──→ Profiling statistique par segment
      │
      ├──→ Analyse du CA par segment
      │
      └──→ Plan marketing + rapport Markdown
```

---

## 10. Métriques clés

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Récence** | (date_ref - dernière_achat).days | Jours depuis le dernier achat — plus petit = meilleur |
| **Fréquence** | nb_transactions par client | Nb d'achats — plus grand = meilleur |
| **Montant** | Σ montant_transaction | CA total client — plus grand = meilleur |
| **Score RFM** | score_R + score_F + score_M | 3–15, Champions proches de 15 |
| **Silhouette** | (b−a) / max(a,b) | 0–1, mesure la qualité du clustering |
| **Inertie** | Σ distances² intra-cluster | Méthode du coude pour choisir k |

---

## 11. Explication des métriques

### RFM — Récence, Fréquence, Montant

L'analyse RFM (Recency, Frequency, Monetary) est un modèle comportemental qui repose sur l'observation que les clients les plus précieux sont ceux qui ont **acheté récemment**, **achètent souvent** et **dépensent le plus**. Chaque dimension est scorée de 1 (mauvais) à 5 (excellent) par quintile pour obtenir un score composite (3–15).

### Score silhouette

Le score silhouette mesure à quel point chaque point est bien classé dans son cluster : `s(i) = (b-a) / max(a,b)` où `a` est la distance intra-cluster et `b` la distance au cluster voisin le plus proche. Un score proche de 1 indique des clusters bien séparés.

### Attribution des labels (Champions → Perdus)

Après clustering, les 5 groupes sont triés par score RFM moyen décroissant et reçoivent automatiquement les labels Champions, Loyaux, Potentiels, À risque, Perdus — garantissant une interprétation stable indépendante de l'ordre aléatoire K-means.

---

## 12. Résultats obtenus

### Sur le dataset simulé (80 clients, 313 transactions)

| Segment | Nb clients | % | Récence moy. | Fréquence moy. | CA moyen |
|---------|-----------|---|-------------|----------------|---------|
| **Champions** | ~16 | 20% | 12 jours | 9.5 achats | 1 250 € |
| **Loyaux** | ~18 | 22% | 45 jours | 5.2 achats | 680 € |
| **Potentiels** | ~14 | 18% | 28 jours | 2.8 achats | 320 € |
| **À risque** | ~16 | 20% | 380 jours | 3.1 achats | 520 € |
| **Perdus** | ~16 | 20% | 1200 jours | 2.4 achats | 280 € |

### Concentration du CA
Les Champions (20% des clients) représentent environ **42% du CA total** — illustration classique du principe de Pareto en CRM.

---

## 13. Valeur métier

### Pour un directeur marketing
- **Personnalisation** : 5 messages différents au lieu d'un message générique → taux conversion x2–3
- **Priorisation budget** : concentrer les remises sur les À risque, pas sur les Champions (qui achètent sans incitation)
- **Alerte précoce** : détecter les Champions qui glissent vers Loyaux avant qu'ils ne partent

### Exemple de ROI concret
Si 20% des clients "À risque" sont réactivés par une offre ciblée à -15% (vs campagne générale à -20%) :
- Économie de remise : 5% × CA segment = gain de marge direct
- Plus de pertinence = moins de désabonnements = LTV préservée

---

## 14. Limites du projet

| Limite | Impact | Mitigation |
|--------|--------|-----------|
| 80 clients seulement | Clusters peu stables | Déployer sur 10 000+ clients |
| RFM statique (une date) | Pas de suivi dans le temps | Rolling window mensuel |
| K-means sensible aux outliers | Un client exceptionnel fausse les clusters | DBSCAN ou isolation avant clustering |
| 3 dimensions RFM uniquement | Ignore les préférences produit | Ajouter catégorie, canal, satisfaction |
| Pas de scoring de propension | Pas de prédiction d'achat futur | Combiner avec modèle de propension |

---

## 15. Améliorations possibles

- **RFM dynamique** : recalcul mensuel automatique + tracking de l'évolution des segments
- **DBSCAN** : clustering sans hypothèse sur le nombre de clusters, robuste aux outliers
- **CLV (Customer Lifetime Value)** : prédire la valeur future de chaque client sur 12 mois
- **Propension à l'achat** : modèle XGBoost sur les transitions de segments
- **A/B testing** : mesurer l'effet réel des campagnes par segment
- **Enrichissement** : données comportementales web, ouvertures d'emails, NPS

---

## 16. Architecture du dépôt

```
customer-marketing-segmentation/
│
├── data_sample/
│   ├── customers.csv          # 80 clients fictifs, 7 attributs
│   ├── transactions.csv       # 313 transactions, 7 colonnes
│   └── schema_reference.md   # Dictionnaire + règles métier
│
├── src/
│   ├── __init__.py
│   ├── rfm_analyzer.py        # Classe RFMAnalyzer (8 méthodes)
│   └── visualization.py      # Classe SegmentationVisualizer (8 figures)
│
├── notebooks/
│   └── 01_customer_segmentation.py  # Script complet 10 sections
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

## 17. README technique

### Installation

```bash
git clone https://github.com/TSAGUE25/customer-marketing-segmentation
cd customer-marketing-segmentation
pip install -r requirements.txt
```

### Exécution

```bash
python notebooks/01_customer_segmentation.py
```

### Utilisation directe

```python
import pandas as pd
from datetime import date
from src.rfm_analyzer import RFMAnalyzer
from src.visualization import SegmentationVisualizer

customers    = pd.read_csv('data_sample/customers.csv')
transactions = pd.read_csv('data_sample/transactions.csv')

analyzer = RFMAnalyzer(customers, transactions, reference_date=date(2024, 12, 4))
clustered = analyzer.cluster(n_clusters=5)

print(analyzer.segment_profile())
print(analyzer.marketing_recommendations())

viz = SegmentationVisualizer(output_dir='figures')
viz.plot_pca_clusters(clustered)
viz.plot_radar_profiles(clustered)
```

---

## 18. Version CV

> *À copier dans la section "Projets" du CV*

**Customer Marketing Segmentation** | Python, scikit-learn, pandas, matplotlib  
Segmentation RFM de 80 clients sur 313 transactions : scoring quintile (R/F/M), K-means (k=5 validé par silhouette), PCA pour visualisation 2D, profiling de 5 segments actionnables (Champions → Perdus), plan marketing personnalisé et rapport Markdown automatisé. Les Champions (20% clients) représentent 42% du CA.

---

## 19. Version entretien

*Question : "Comment approcheriez-vous une problématique de segmentation client ?"*

> "Je commence par l'analyse RFM — Récence, Fréquence, Montant — car c'est la méthode la plus éprouvée en CRM et elle est directement interprétable par les équipes marketing, sans boîte noire.
>
> Concrètement : j'agrège l'historique transactionnel par client, je calcule les 3 métriques, je les score de 1 à 5 par quintile (la récence est inversée car plus c'est récent, mieux c'est), puis j'applique K-means sur les valeurs normalisées.
>
> Pour choisir le k, j'utilise à la fois la méthode du coude et le score silhouette — les deux sont complémentaires. Ici, k=5 donne le meilleur silhouette (~0.42).
>
> Le point clé : après le clustering, je trie les 5 groupes par score RFM moyen décroissant et je leur attribue des noms métier (Champions, Loyaux, Potentiels, À risque, Perdus). Ça transforme un résultat technique en outil opérationnel que le marketing peut utiliser directement."

---

## 20. Version portfolio

Ce projet illustre la maîtrise du cycle complet d'un projet CRM analytique : de la donnée transactionnelle brute jusqu'au plan d'action marketing personnalisé par segment. Il démontre la capacité à choisir des méthodes adaptées (RFM pour l'interprétabilité, K-means pour l'efficacité) et à produire des livrables utilisables par des équipes non techniques.

**Adapté pour :** e-commerce, retail, banque, assurance, télécoms, tout secteur B2C avec historique client.

---

## 21. Post LinkedIn

> **La segmentation RFM m'a appris quelque chose de contre-intuitif sur le marketing**
>
> On croit naturellement qu'il faut offrir des remises aux meilleurs clients pour les fidéliser. C'est souvent une erreur.
>
> En analysant un portefeuille de 80 clients (simulés), voici ce que j'ai trouvé :
>
> Les **Champions** (20% des clients) génèrent **42% du CA** — et ils achètent SANS promotion.  
> Les **À risque** ont besoin d'une offre personnalisée AVANT de partir, pas après.  
> Les **Perdus** coûtent plus cher à réactiver qu'à acquérir de nouveaux clients.
>
> La segmentation RFM + K-means permet de cibler précisément chaque groupe :  
> → Récence (jours depuis le dernier achat)  
> → Fréquence (nombre de transactions)  
> → Montant (CA total)
>
> Résultat : 5 segments actionnables, 8 visualisations, plan marketing automatisé.  
> Code sur GitHub  
> #DataScience #CRM #Marketing #Python #Segmentation #KMeans #RFM #Portfolio

---

## 22. Questions d'entretien

**Q1 : Pourquoi K-means plutôt que DBSCAN ou clustering hiérarchique ?**  
K-means est rapide, scalable (fonctionne sur des millions de clients), et ses résultats sont facilement interprétables. DBSCAN serait meilleur si les clusters ont des formes irrégulières ou si on a des outliers marqués. Le hiérarchique est utile pour explorer, mais trop lent en production.

**Q2 : Comment gérer la sensibilité de K-means aux outliers ?**  
Identifier et traiter les outliers avant le clustering : soit les isoler dans un segment "VIP exceptionnel", soit les winsoriser (plafonner à Q3+3×IQR). Sinon un client à 50 000€ de CA fausse le centroïde du cluster le plus riche.

**Q3 : Pourquoi scorer par quintile plutôt que de normaliser directement ?**  
Le scoring par quintile est robuste : il ne suppose pas de distribution normale et transforme des distributions très asymétriques (montants) en scores uniformes 1–5. La normalisation brute (StandardScaler) conserve les outliers qui peuvent dominer les résultats K-means.

**Q4 : Comment valider la qualité des segments en conditions réelles ?**  
A/B test : appliquer des campagnes différenciées par segment pendant 3 mois et mesurer le taux de conversion, le panier moyen et le taux de rétention par segment vs groupe contrôle.

**Q5 : Quelle est la différence entre segmentation et scoring de propension ?**  
La segmentation (RFM/K-means) décrit ce que les clients ONT FAIT. Le scoring de propension prédit ce qu'ils FERONT (probabilité d'achat dans les 30 prochains jours). Les deux sont complémentaires : segmentation pour cibler, propension pour prioriser.

---

## 23. Compétences démontrées

- **pandas** — groupby multi-agrégation, pd.qcut, merge, pivot
- **scikit-learn** — KMeans, StandardScaler, PCA, silhouette_score
- **matplotlib / seaborn** — scatter, boxplot, heatmap, radar, pie, barh, timeline
- **Statistiques** — quintiles, silhouette, inertie, méthode du coude
- **Marketing analytique** — RFM, segmentation comportementale, CLV
- **Python OOP** — classes `RFMAnalyzer` et `SegmentationVisualizer`
- **Reporting** — génération Markdown programmatique

---

## 24. Tableau compétences / preuves

| Compétence | Preuve | Fichier |
|-----------|--------|---------|
| Calcul RFM | Méthode `compute_rfm()` | `src/rfm_analyzer.py` |
| Scoring quintile | Méthode `compute_rfm_scores()` | `src/rfm_analyzer.py` |
| Choix k optimal | Méthode `find_optimal_k()` | `src/rfm_analyzer.py` |
| K-means + labels | Méthode `cluster()` | `src/rfm_analyzer.py` |
| PCA 2D | Intégré dans `cluster()` | `src/rfm_analyzer.py` |
| Profiling segments | Méthode `segment_profile()` | `src/rfm_analyzer.py` |
| Plan marketing | Méthode `marketing_recommendations()` | `src/rfm_analyzer.py` |
| 8 visualisations | Classe `SegmentationVisualizer` | `src/visualization.py` |
| Script pipeline | 10 sections commentées | `notebooks/01_customer_segmentation.py` |

---

## 25. Conseils GitHub

**Description :** "Segmentation marketing RFM + K-means : 5 segments actionnables (Champions → Perdus), plan marketing personnalisé. Portfolio Data Science."

**Topics :** `python` `data-science` `marketing` `rfm` `kmeans` `customer-segmentation` `pandas` `scikit-learn` `crm` `portfolio`

**Projets connexes :**
| Projet | Lien |
|--------|------|
| Data Quality Audit | [data-quality-audit-framework](https://github.com/TSAGUE25/data-quality-audit-framework) |
| Building Energy Analytics | [building-energy-efficiency-analytics](https://github.com/TSAGUE25/building-energy-efficiency-analytics) |
| Bank Churn Prediction | [bank-customer-churn-prediction](https://github.com/TSAGUE25/bank-customer-churn-prediction) |


## Contributors

| Nom | Role | GitHub |
|-----|------|--------|
| **TSAGUE Emmanuel** | Data Scientist - auteur principal | [@TSAGUE25](https://github.com/TSAGUE25) |

---

*Auteur : Emmanuel TSAGUE — Data Scientist / Data Analyst*  
*Formation : DataScientest | Domaines : Commerce · Finance · Energie · Performance opérationnelle*  
*Contact : emmatsague@yahoo.fr*  
*Données : entièrement simulées — aucune donnée réelle ou confidentielle*
