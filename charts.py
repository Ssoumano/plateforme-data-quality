"""Graphiques Plotly interactifs de la plateforme."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY = "#667eea"
SEVERITY_COLORS = {"critical": "#F44336", "high": "#FF9800", "medium": "#FFC107", "low": "#4CAF50"}


def _style(fig, height=None):
    """Style commun : fond transparent (thème clair/sombre), marges réduites."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        title_font_size=16,
    )
    return fig


def _severity(pct: float) -> str:
    if pct > 50:
        return "critical"
    if pct > 30:
        return "high"
    if pct > 10:
        return "medium"
    return "low"


def create_missing_chart(profil, top_n=10):
    """Barres horizontales des colonnes avec le plus de valeurs manquantes."""
    mp = pd.Series(profil["missing_pct"])
    mp = mp[mp > 0].sort_values(ascending=False).head(top_n)
    if mp.empty:
        return None

    fig = go.Figure(go.Bar(
        x=mp.values, y=[str(c) for c in mp.index], orientation="h",
        marker_color=[SEVERITY_COLORS[_severity(v)] for v in mp.values],
        text=[f"{v:.1f} %" for v in mp.values], textposition="outside",
        hovertemplate="%{y} : %{x:.1f} %<extra></extra>",
    ))
    fig.update_xaxes(title="Valeurs manquantes (%)", range=[0, 125])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"Top {len(mp)} colonnes avec valeurs manquantes")
    return _style(fig, height=max(300, 40 * len(mp) + 100))


def create_types_chart(col_types):
    """Donut de la répartition des types de colonnes."""
    s = pd.Series(col_types).value_counts()
    fig = go.Figure(go.Pie(
        labels=s.index.tolist(), values=s.values.tolist(), hole=0.5,
        marker=dict(colors=px.colors.qualitative.Set2),
        hovertemplate="%{label} : %{value} colonne(s)<extra></extra>",
    ))
    fig.update_layout(title="Types de colonnes")
    return _style(fig, height=380)


def create_outliers_chart(profil, top_n=15):
    """Barres du nombre d'outliers (IQR) par colonne numérique."""
    outliers = {k: v for k, v in profil.get("outliers", {}).items() if v > 0}
    if not outliers:
        return None
    s = pd.Series(outliers).sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=s.values, y=[str(c) for c in s.index], orientation="h",
        marker_color="#EF5350", text=s.values, textposition="outside",
        hovertemplate="%{y} : %{x} outlier(s)<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Nombre d'outliers")
    fig.update_layout(title="Outliers par colonne (méthode IQR)")
    return _style(fig, height=max(250, 40 * len(s) + 100))


def create_correlation_chart(corr):
    """Heatmap interactive de la matrice de corrélation."""
    fig = px.imshow(
        corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        text_auto=".2f" if len(corr) <= 12 else False, aspect="auto",
    )
    fig.update_layout(title="Corrélations entre variables numériques")
    return _style(fig, height=min(800, max(350, 60 * len(corr))))


def create_numeric_charts(series, name):
    """Histogramme et boxplot d'une colonne numérique."""
    clean = series.dropna()
    fig_dist = px.histogram(clean, nbins=30, color_discrete_sequence=[PRIMARY])
    fig_dist.update_layout(title=f"Distribution de {name}", showlegend=False,
                           xaxis_title=str(name), yaxis_title="Fréquence")
    fig_box = px.box(clean, points="outliers", color_discrete_sequence=[PRIMARY])
    fig_box.update_layout(title=f"Boxplot de {name}", showlegend=False, yaxis_title=str(name))
    return _style(fig_dist, 380), _style(fig_box, 380)


def create_category_chart(series, name, top_n=20):
    """Top valeurs d'une colonne non numérique."""
    value_counts = series.value_counts().head(top_n)
    fig = go.Figure(go.Bar(
        x=value_counts.values, y=[str(v) for v in value_counts.index], orientation="h",
        marker_color=PRIMARY, hovertemplate="%{y} : %{x}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Fréquence")
    fig.update_layout(title=f"Top {len(value_counts)} valeurs de {name}")
    return _style(fig, height=max(300, 28 * len(value_counts) + 100)), value_counts
