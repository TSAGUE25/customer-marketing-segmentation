import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

SEGMENT_COLORS = {
    'Champions': '#00C851',
    'Loyaux': '#33b5e5',
    'Potentiels': '#ffbb33',
    'A_risque': '#ff8800',
    'Perdus': '#ff4444',
}
SEGMENT_ORDER = ['Champions', 'Loyaux', 'Potentiels', 'A_risque', 'Perdus']


class SegmentationVisualizer:
    """Visualisations pour la segmentation marketing RFM."""

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})

    def _save(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure sauvegardée : {path}")
        return path

    def _colors(self, segments):
        return [SEGMENT_COLORS.get(s, '#999999') for s in segments]

    # ------------------------------------------------------------------ #
    # Figure 1 : Méthode du coude + silhouette
    # ------------------------------------------------------------------ #

    def plot_elbow(self, k_range, inertias, silhouettes):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax1.set_xlabel('Nombre de clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertie (WCSS)', fontsize=12)
        ax1.set_title('Méthode du coude', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax2.plot(k_range, silhouettes, 's-', color='crimson', linewidth=2, markersize=8)
        best_k = k_range[np.argmax(silhouettes)]
        ax2.axvline(best_k, color='crimson', linestyle='--', alpha=0.5, label=f'k optimal = {best_k}')
        ax2.set_xlabel('Nombre de clusters (k)', fontsize=12)
        ax2.set_ylabel('Score silhouette', fontsize=12)
        ax2.set_title('Score silhouette', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        fig.suptitle('Choix du nombre de segments K-means', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'fig1_choix_k_clusters.png')

    # ------------------------------------------------------------------ #
    # Figure 2 : Scatter PCA 2D des segments
    # ------------------------------------------------------------------ #

    def plot_pca_clusters(self, df):
        fig, ax = plt.subplots(figsize=(10, 7))
        segments = df['segment'].unique()
        for seg in SEGMENT_ORDER:
            if seg not in segments:
                continue
            sub = df[df['segment'] == seg]
            ax.scatter(sub['pca_1'], sub['pca_2'],
                       c=SEGMENT_COLORS.get(seg, '#999'), s=60, alpha=0.7,
                       label=f"{seg} (n={len(sub)})", edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Composante principale 1', fontsize=12)
        ax.set_ylabel('Composante principale 2', fontsize=12)
        ax.set_title('Segmentation clients — Vue PCA (RFM normalisé)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.25)
        return self._save(fig, 'fig2_pca_segments.png')

    # ------------------------------------------------------------------ #
    # Figure 3 : Boxplot RFM par segment
    # ------------------------------------------------------------------ #

    def plot_rfm_boxplots(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = [
            ('recence', 'Récence (jours)', True),
            ('frequence', 'Fréquence (nb achats)', False),
            ('montant_total', 'CA total (€)', False),
        ]
        order = [s for s in SEGMENT_ORDER if s in df['segment'].unique()]
        palette = {s: SEGMENT_COLORS.get(s, '#999') for s in order}
        for ax, (col, label, reverse) in zip(axes, metrics):
            sns.boxplot(data=df, x='segment', y=col, order=order,
                        palette=palette, ax=ax, width=0.6)
            ax.set_xlabel('')
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('Distribution RFM par segment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'fig3_rfm_par_segment.png')

    # ------------------------------------------------------------------ #
    # Figure 4 : Répartition clients et CA par segment (donut)
    # ------------------------------------------------------------------ #

    def plot_segment_distribution(self, profile_df, revenue_df):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        order = [s for s in SEGMENT_ORDER if s in profile_df.index]
        colors = self._colors(order)

        # Donut clients
        ax1.pie(profile_df.loc[order, 'nb_clients'], labels=order, colors=colors,
                autopct='%1.0f%%', startangle=90,
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
        ax1.set_title('Répartition des clients', fontsize=13, fontweight='bold')

        # Barh CA
        rev = revenue_df.reindex([s for s in SEGMENT_ORDER if s in revenue_df.index])
        bars = ax2.barh(rev.index, rev['ca_total'], color=self._colors(rev.index),
                        edgecolor='white', height=0.6)
        for bar, (_, row) in zip(bars, rev.iterrows()):
            ax2.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                     f"{row['ca_total']:,.0f} € ({row['pct_ca']:.0f}%)",
                     va='center', fontsize=9)
        ax2.set_xlabel('CA total (€)', fontsize=12)
        ax2.set_title('CA total par segment', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        fig.suptitle('Valeur des segments client', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'fig4_valeur_segments.png')

    # ------------------------------------------------------------------ #
    # Figure 5 : Heatmap scores RFM moyens par segment
    # ------------------------------------------------------------------ #

    def plot_rfm_heatmap(self, df):
        order = [s for s in SEGMENT_ORDER if s in df['segment'].unique()]
        pivot = df.groupby('segment')[['score_R', 'score_F', 'score_M']].mean().reindex(order)
        pivot.columns = ['Score Récence', 'Score Fréquence', 'Score Montant']
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=1, vmax=5, linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'Score (1=faible, 5=élevé)'})
        ax.set_title('Scores RFM moyens par segment', fontsize=14, fontweight='bold')
        ax.set_ylabel('')
        plt.tight_layout()
        return self._save(fig, 'fig5_heatmap_rfm.png')

    # ------------------------------------------------------------------ #
    # Figure 6 : Évolution des achats dans le temps
    # ------------------------------------------------------------------ #

    def plot_purchase_timeline(self, transactions_df, clustered_df):
        df = transactions_df.merge(
            clustered_df[['id_client', 'segment']], on='id_client', how='left'
        )
        df['annee_mois'] = df['date_transaction'].dt.to_period('M')
        monthly = df.groupby(['annee_mois', 'segment'])['montant'].sum().reset_index()
        monthly['annee_mois_dt'] = monthly['annee_mois'].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(12, 6))
        for seg in SEGMENT_ORDER:
            sub = monthly[monthly['segment'] == seg].sort_values('annee_mois_dt')
            if sub.empty:
                continue
            ax.plot(sub['annee_mois_dt'], sub['montant'],
                    marker='o', markersize=4, linewidth=1.5,
                    color=SEGMENT_COLORS.get(seg, '#999'), label=seg, alpha=0.85)
        ax.set_xlabel('Période', fontsize=12)
        ax.set_ylabel('CA mensuel (€)', fontsize=12)
        ax.set_title('Évolution du CA mensuel par segment', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        return self._save(fig, 'fig6_ca_mensuel_segments.png')

    # ------------------------------------------------------------------ #
    # Figure 7 : Radar chart profil par segment
    # ------------------------------------------------------------------ #

    def plot_radar_profiles(self, df):
        order = [s for s in SEGMENT_ORDER if s in df['segment'].unique()]
        metrics = ['score_R', 'score_F', 'score_M']
        labels = ['Récence', 'Fréquence', 'Montant']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for seg in order:
            vals = df[df['segment'] == seg][metrics].mean().values.tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, color=SEGMENT_COLORS.get(seg, '#999'), label=seg)
            ax.fill(angles, vals, alpha=0.08, color=SEGMENT_COLORS.get(seg, '#999'))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_title('Profil RFM moyen par segment', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return self._save(fig, 'fig7_radar_profils.png')

    # ------------------------------------------------------------------ #
    # Figure 8 : Recommandations marketing (tableau visuel)
    # ------------------------------------------------------------------ #

    def plot_marketing_plan(self, recs_df):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('off')
        order = [s for s in SEGMENT_ORDER if s in recs_df['segment'].values]
        recs_df = recs_df.set_index('segment').reindex(order).reset_index()
        table_data = [
            [row['segment'], row['nb_clients'], row['description'], row['action_recommandee']]
            for _, row in recs_df.iterrows()
        ]
        col_labels = ['Segment', 'Nb clients', 'Profil', 'Action recommandée']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='left', loc='center', colWidths=[0.15, 0.08, 0.35, 0.42])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i, seg in enumerate(recs_df['segment'], start=1):
            color = SEGMENT_COLORS.get(seg, '#ffffff')
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(color + '33')
        ax.set_title('Plan d\'action marketing par segment', fontsize=14,
                     fontweight='bold', pad=20)
        plt.tight_layout()
        return self._save(fig, 'fig8_plan_marketing.png')
