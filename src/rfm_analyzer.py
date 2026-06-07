import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


SEGMENT_LABELS = {
    'Champions': {
        'description': 'Achat récent, fréquent, montant élevé',
        'action': 'Récompenser, ambassadeurs, early access',
        'color': '#00C851'
    },
    'Loyaux': {
        'description': 'Achètent régulièrement, bon panier moyen',
        'action': 'Programme fidélité, ventes croisées',
        'color': '#33b5e5'
    },
    'Potentiels': {
        'description': 'Récents mais peu fréquents',
        'action': 'Onboarding, offres découverte',
        'color': '#ffbb33'
    },
    'A_risque': {
        'description': 'Bons clients, achats moins récents',
        'action': 'Campagne win-back, offre spéciale',
        'color': '#ff8800'
    },
    'Perdus': {
        'description': 'Inactifs depuis longtemps',
        'action': 'Réactivation ciblée ou abandon',
        'color': '#ff4444'
    },
}


class RFMAnalyzer:
    """Segmentation clients par score RFM et K-means clustering."""

    def __init__(self, customers_df, transactions_df, reference_date=None):
        self.customers = customers_df.copy()
        self.transactions = transactions_df.copy()
        self.reference_date = reference_date or date.today()
        self.transactions['date_transaction'] = pd.to_datetime(
            self.transactions['date_transaction']
        )
        self._rfm = None
        self._clustered = None
        self.scaler = StandardScaler()
        self.kmeans = None

    # ------------------------------------------------------------------ #
    # 1. Calcul RFM brut
    # ------------------------------------------------------------------ #

    def compute_rfm(self):
        """Calcule Récence (jours), Fréquence (nb transactions), Montant (total €)."""
        ref = pd.Timestamp(self.reference_date)
        rfm = (
            self.transactions
            .groupby('id_client')
            .agg(
                derniere_transaction=('date_transaction', 'max'),
                frequence=('id_transaction', 'count'),
                montant_total=('montant', 'sum'),
                montant_moyen=('montant', 'mean'),
                satisfaction_moy=('satisfaction', 'mean'),
            )
            .reset_index()
        )
        rfm['recence'] = (ref - rfm['derniere_transaction']).dt.days
        rfm = rfm.merge(
            self.customers[['id_client', 'age', 'genre', 'ville', 'region',
                             'canal_acquisition', 'categorie_principale']],
            on='id_client', how='left'
        )
        self._rfm = rfm
        return rfm

    def get_rfm(self):
        if self._rfm is None:
            self.compute_rfm()
        return self._rfm

    # ------------------------------------------------------------------ #
    # 2. Scoring RFM (1-5 par quintile)
    # ------------------------------------------------------------------ #

    def compute_rfm_scores(self):
        """Attribue des scores 1-5 à chaque dimension RFM."""
        rfm = self.get_rfm().copy()
        # Récence : plus faible = meilleur → score inversé
        rfm['score_R'] = pd.qcut(rfm['recence'], q=5,
                                  labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
        # Fréquence : plus élevée = meilleur
        rfm['score_F'] = pd.qcut(rfm['frequence'].rank(method='first'), q=5,
                                  labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
        # Montant : plus élevé = meilleur
        rfm['score_M'] = pd.qcut(rfm['montant_total'].rank(method='first'), q=5,
                                  labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
        rfm['score_rfm'] = rfm['score_R'] + rfm['score_F'] + rfm['score_M']
        self._rfm = rfm
        return rfm

    # ------------------------------------------------------------------ #
    # 3. K-means clustering
    # ------------------------------------------------------------------ #

    def find_optimal_k(self, k_range=range(2, 8)):
        """Méthode du coude + silhouette pour trouver le k optimal."""
        rfm = self.get_rfm()
        X = self.scaler.fit_transform(rfm[['recence', 'frequence', 'montant_total']])
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, km.labels_))
        return list(k_range), inertias, silhouettes

    def cluster(self, n_clusters=5):
        """Applique K-means avec n_clusters et assigne un segment à chaque client."""
        rfm = self.compute_rfm_scores()
        X = self.scaler.fit_transform(rfm[['recence', 'frequence', 'montant_total']])
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['cluster'] = self.kmeans.fit_predict(X)

        # Attribution automatique des labels par profil RFM moyen du cluster
        cluster_profile = rfm.groupby('cluster')[['score_R', 'score_F', 'score_M']].mean()
        cluster_profile['score_total'] = cluster_profile.sum(axis=1)
        sorted_clusters = cluster_profile['score_total'].sort_values(ascending=False).index.tolist()

        label_map = {}
        segment_names = ['Champions', 'Loyaux', 'Potentiels', 'A_risque', 'Perdus']
        for i, cluster_id in enumerate(sorted_clusters):
            label_map[cluster_id] = segment_names[i] if i < len(segment_names) else f'Segment_{i}'
        rfm['segment'] = rfm['cluster'].map(label_map)

        # PCA 2D pour visualisation
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        rfm['pca_1'] = coords[:, 0]
        rfm['pca_2'] = coords[:, 1]
        rfm['pca_variance'] = [pca.explained_variance_ratio_] * len(rfm)

        self._clustered = rfm
        self._label_map = label_map
        self._pca = pca
        return rfm

    def get_clustered(self):
        if self._clustered is None:
            self.cluster()
        return self._clustered

    # ------------------------------------------------------------------ #
    # 4. Profiling des segments
    # ------------------------------------------------------------------ #

    def segment_profile(self):
        """Statistiques clés par segment."""
        df = self.get_clustered()
        profile = (
            df.groupby('segment')
            .agg(
                nb_clients=('id_client', 'count'),
                recence_moy=('recence', 'mean'),
                frequence_moy=('frequence', 'mean'),
                montant_total_moy=('montant_total', 'mean'),
                montant_moyen_moy=('montant_moyen', 'mean'),
                score_rfm_moy=('score_rfm', 'mean'),
                satisfaction_moy=('satisfaction_moy', 'mean'),
            )
            .round(1)
        )
        profile['pct_clients'] = (profile['nb_clients'] / profile['nb_clients'].sum() * 100).round(1)
        return profile

    def segment_revenue(self):
        """CA total et moyen par segment."""
        df = self.get_clustered()
        rev = df.groupby('segment').agg(
            nb_clients=('id_client', 'count'),
            ca_total=('montant_total', 'sum'),
            ca_moyen=('montant_total', 'mean'),
        ).round(0)
        rev['pct_ca'] = (rev['ca_total'] / rev['ca_total'].sum() * 100).round(1)
        return rev.sort_values('ca_total', ascending=False)

    # ------------------------------------------------------------------ #
    # 5. Recommandations marketing
    # ------------------------------------------------------------------ #

    def marketing_recommendations(self):
        """Retourne les recommandations d'action par segment."""
        df = self.get_clustered()
        recs = []
        for seg, info in SEGMENT_LABELS.items():
            count = (df['segment'] == seg).sum()
            if count > 0:
                recs.append({
                    'segment': seg,
                    'nb_clients': count,
                    'description': info['description'],
                    'action_recommandee': info['action'],
                })
        return pd.DataFrame(recs)

    # ------------------------------------------------------------------ #
    # 6. Synthèse
    # ------------------------------------------------------------------ #

    def summary(self):
        df = self.get_clustered()
        return {
            'nb_clients_total': len(df),
            'nb_segments': df['segment'].nunique(),
            'ca_total': round(df['montant_total'].sum(), 0),
            'ca_moyen_client': round(df['montant_total'].mean(), 0),
            'recence_moyenne_jours': round(df['recence'].mean(), 0),
            'frequence_moyenne': round(df['frequence'].mean(), 1),
            'champions_pct': round((df['segment'] == 'Champions').mean() * 100, 1),
            'perdus_pct': round((df['segment'] == 'Perdus').mean() * 100, 1),
            'segment_distribution': df['segment'].value_counts().to_dict(),
        }
