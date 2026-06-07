# Méthodologie — Customer Marketing Segmentation

## 1. Analyse RFM

### Définition des trois dimensions

| Dimension | Signification | Calcul | Signe |
|-----------|---------------|--------|-------|
| **Récence (R)** | Depuis combien de temps n'a-t-il pas acheté ? | (date_ref - dernière transaction).days | Plus petit = meilleur |
| **Fréquence (F)** | Combien de fois a-t-il acheté ? | count(transactions) | Plus grand = meilleur |
| **Montant (M)** | Combien a-t-il dépensé au total ? | sum(montant) | Plus grand = meilleur |

### Scoring par quintile

Chaque dimension est divisée en 5 quintiles (20% des clients chacun) et reçoit un score de 1 à 5.  
La récence est **inversée** : un client qui a acheté il y a 3 jours reçoit un score R=5, pas un score R=1.

```
Score 5 = top 20% du portefeuille
Score 1 = bottom 20% du portefeuille
Score RFM total = R + F + M ∈ [3, 15]
```

---

## 2. Choix du nombre de clusters (k)

### Méthode du coude (Elbow)
Tracer l'inertie (WCSS = somme des distances² intra-cluster) en fonction de k. Le "coude" indique le k au-delà duquel gagner en précision ne vaut plus le coût de complexité.

### Score silhouette
`s(i) = (b - a) / max(a, b)` où :
- `a` = distance moyenne au sein du même cluster (cohésion)
- `b` = distance moyenne au cluster voisin le plus proche (séparation)
- Score ∈ [-1, 1] ; proche de 1 = bon clustering

Le k retenu est celui qui maximise le score silhouette tout en présentant un coude visible sur l'inertie.

---

## 3. K-means clustering

### Principe
K-means minimise la somme des distances euclidiennes entre chaque point et le centroïde de son cluster :
```
min Σ_k Σ_{x ∈ C_k} ||x - μ_k||²
```

### Prétraitement obligatoire
K-means est **sensible aux échelles**. Un Montant de 1 500 € dominerait une Récence de 30 jours sans normalisation. On applique `StandardScaler` pour centrer-réduire chaque variable.

### Attribution des labels
Après clustering, les 5 groupes sont triés par score RFM moyen décroissant et reçoivent les labels dans cet ordre : Champions, Loyaux, Potentiels, À_risque, Perdus.

---

## 4. Visualisation PCA

La PCA (Analyse en Composantes Principales) projette les 3 dimensions RFM normalisées en 2 dimensions, permettant de visualiser les clusters. Les deux premières composantes expliquent généralement 85-95% de la variance totale.
