"""Graphiques matplotlib/seaborn de la plateforme."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_missing_chart(profil, top_n=10):
    """Crée un graphique des colonnes avec le plus de valeurs manquantes."""
    mp = profil["missing_pct"]
    if isinstance(mp, dict):
        mp = pd.Series(mp)
    
    mp = mp[mp > 0].sort_values(ascending=False).head(top_n)
    
    if mp.empty or mp.sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(mp) * 0.4)))
    
    # Couleurs basées sur le niveau de sévérité
    colors_list = [
        '#F44336' if x > 50 else 
        '#FF9800' if x > 30 else 
        '#FFC107' if x > 10 else 
        '#4CAF50' 
        for x in mp.values
    ]
    
    mp.plot(kind='barh', ax=ax, color=colors_list, edgecolor='white', linewidth=1.5)
    ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11, fontweight='bold')
    ax.set_title(f"Top {len(mp)} colonnes avec valeurs manquantes", fontsize=13, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(mp.values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig


def create_types_chart(col_types):
    """Crée un pie chart de la distribution des types de colonnes."""
    s = pd.Series(col_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FFC107', '#F44336', '#9C27B0', '#00BCD4']
    explode = [0.05] * len(s)
    
    wedges, texts, autotexts = ax.pie(
        s,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[:len(s)],
        explode=explode,
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    ax.legend(wedges, s.index, title="Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title("Distribution des types de colonnes", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_outliers_heatmap(profil):
    """Crée une heatmap des outliers par colonne."""
    outliers = profil.get("outliers", {})
    
    if not outliers or all(v == 0 for v in outliers.values()):
        return None
    
    # Filtrer les colonnes avec outliers
    outliers_filtered = {k: v for k, v in outliers.items() if v > 0}
    
    if not outliers_filtered:
        return None
    
    outlier_df = pd.DataFrame(
        list(outliers_filtered.items()), 
        columns=["Colonne", "Nb Outliers"]
    ).set_index("Colonne")
    
    fig, ax = plt.subplots(figsize=(10, max(3, len(outlier_df) * 0.5)))
    
    sns.heatmap(
        outlier_df,
        annot=True,
        fmt="d",
        cmap="Reds",
        ax=ax,
        cbar_kws={"label": "Nombre d'outliers"},
        linewidths=1,
        linecolor='white'
    )
    
    ax.set_title("Détection d'outliers (méthode IQR)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    return fig


def create_correlation_chart(corr):
    """Heatmap de la matrice de corrélation."""
    size = min(14, max(6, len(corr) * 0.6))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, annot=len(corr) <= 15, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Corrélations entre variables numériques", fontsize=14, fontweight='bold', pad=15)
    return fig


def create_numeric_charts(series, name):
    """Histogramme et boxplot d'une colonne numérique."""
    fig_dist, ax = plt.subplots(figsize=(8, 5))
    series.dropna().hist(bins=30, ax=ax, color="#667eea", edgecolor="white")
    ax.set_xlabel(name)
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution de {name}")
    ax.grid(alpha=0.3)

    fig_box, ax = plt.subplots(figsize=(8, 5))
    series.dropna().plot.box(ax=ax, color=dict(boxes='#667eea', whiskers='#764ba2', medians='#F44336'))
    ax.set_ylabel(name)
    ax.set_title(f"Boxplot de {name}")
    ax.grid(axis='y', alpha=0.3)
    return fig_dist, fig_box


def create_category_chart(series, name, top_n=20):
    """Top valeurs d'une colonne non numérique."""
    value_counts = series.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts.plot(kind='barh', ax=ax, color='#667eea')
    ax.set_xlabel("Fréquence")
    ax.set_title(f"Top {top_n} valeurs de {name}")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    return fig, value_counts
