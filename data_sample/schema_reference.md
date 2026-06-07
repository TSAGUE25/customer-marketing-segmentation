# Schéma de référence — Customer Marketing Segmentation

> Données entièrement simulées à des fins pédagogiques.

## Table 1 : customers.csv (80 lignes)

| Colonne | Type | Valeurs |
|---------|------|---------|
| `id_client` | string | C001–C080, unique, non nul |
| `age` | integer | 22–67 |
| `genre` | string | F, M |
| `ville` | string | 17 villes françaises |
| `region` | string | 13 régions |
| `date_inscription` | date | 2015-01-01 à 2022-12-31 |
| `canal_acquisition` | string | Web, Magasin, Email |
| `categorie_principale` | string | Mode, Electronique, Beauté, Sport, Maison, Alimentaire |

## Table 2 : transactions.csv (313 lignes)

| Colonne | Type | Valeurs |
|---------|------|---------|
| `id_transaction` | string | T0001–T0313, unique |
| `id_client` | string | FK → customers.id_client |
| `date_transaction` | date | 2015-01-01 à 2024-12-03 |
| `montant` | float | 29.50 à 1500.00 € |
| `categorie` | string | Mode, Electronique, Beauté, Sport, Maison, Alimentaire |
| `canal` | string | Web, Magasin, Email |
| `satisfaction` | integer | 3, 4, 5 |

## Variables RFM calculées

| Variable | Formule | Unité |
|----------|---------|-------|
| `recence` | (date_ref - max(date_transaction)).days | jours |
| `frequence` | count(id_transaction) par client | nb |
| `montant_total` | sum(montant) par client | € |
| `score_R` | pd.qcut(recence, q=5, labels=[5,4,3,2,1]) | 1–5 |
| `score_F` | pd.qcut(frequence.rank, q=5, labels=[1,2,3,4,5]) | 1–5 |
| `score_M` | pd.qcut(montant_total.rank, q=5, labels=[1,2,3,4,5]) | 1–5 |
| `score_rfm` | score_R + score_F + score_M | 3–15 |
| `segment` | label K-means par score RFM décroissant | Champions/Loyaux/Potentiels/A_risque/Perdus |
