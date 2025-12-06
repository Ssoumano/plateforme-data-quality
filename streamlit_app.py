# streamlit_app.py - PARTIE 1/2
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import cm
from PIL import Image
import base64
from datetime import datetime


# ----------------------------------------------------
# 🔧 FIX : Nettoyage du texte OpenAI et HTML
# ----------------------------------------------------
def clean_ai_text(text: str) -> str:
    """Nettoie le texte OpenAI et échappe les caractères HTML"""
    text = text.replace("`", "")
    text = text.replace("* *", " ")
    text = text.replace("** **", " ")
    text = text.replace("• •", "• ")
    text = text.replace("\u200b", "")
    return text


def escape_for_reportlab(text: str) -> str:
    """Échappe le texte pour ReportLab"""
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    text = text.replace('&lt;br/&gt;', '<br/>')
    text = re.sub(r'<b>\s*</b>', '', text)
    text = re.sub(r'<i>\s*</i>', '', text)
    return text


# CONFIG API OPENAI
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    OPENAI_API_KEY = None
    client = None


# ----------------------------------------------------
# Import: separator detection & load
# ----------------------------------------------------
def detect_separator(uploaded_file_bytes: bytes) -> str:
    sample = uploaded_file_bytes[:4096].decode(errors='ignore')
    for sep in [';', ',', '\t', '|']:
        if sep in sample:
            return sep
    return ','


@st.cache_data
def load_dataframe(uploaded_file_name: str, uploaded_file_bytes: bytes):
    """Load dataframe with caching for performance"""
    name = uploaded_file_name.lower()
    if name.endswith('.csv'):
        sep = detect_separator(uploaded_file_bytes)
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), sep=sep, encoding='utf-8', engine='python')
    elif name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(uploaded_file_bytes))
    else:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding='utf-8')


# ----------------------------------------------------
# Détection automatique des types de colonnes
# ----------------------------------------------------
@st.cache_data
def detect_column_types(df_dict):
    """Detect column types automatically"""
    df = pd.DataFrame(df_dict)
    types = {}
    
    for col in df.columns:
        if df[col].isna().all():
            types[col] = 'empty'
            continue
        
        try:
            if df[col].dtype == 'object':
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.8:
                    types[col] = 'datetime'
                    continue
        except:
            pass
        
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:
                types[col] = 'categorical'
            else:
                types[col] = 'numeric'
        else:
            nunique = df[col].nunique()
            if nunique == 2:
                types[col] = 'binary'
            elif nunique / len(df) < 0.05:
                types[col] = 'categorical'
            else:
                types[col] = 'text'
    
    return types


# ----------------------------------------------------
# Data Profiling avec détection de cohérence
# ----------------------------------------------------
@st.cache_data
def profile_data_quality(df_dict) -> dict:
    """Profile data quality with caching and coherence checks"""
    df = pd.DataFrame(df_dict)
    
    profil = {}
    profil['rows'] = int(df.shape[0])
    profil['cols'] = int(df.shape[1])
    profil['missing_count'] = df.isna().sum().to_dict()
    profil['missing_pct'] = (df.isna().mean() * 100).round(2).to_dict()
    profil['dtypes'] = df.dtypes.astype(str).to_dict()
    profil['constant_columns'] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    profil['empty_columns'] = [c for c in df.columns if df[c].dropna().shape[0] == 0]
    profil['duplicate_rows'] = int(df.duplicated().sum())

    numeric = df.select_dtypes(include=[np.number])
    
    if len(numeric.columns) > 0:
        profil['numeric_stats'] = numeric.describe().T.to_dict()
    else:
        profil['numeric_stats'] = {}

    # Outliers via IQR
    outliers = {}
    for col in numeric.columns:
        x = df[col].dropna()
        if x.empty:
            outliers[col] = 0
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[col] = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    profil['outliers'] = outliers

    # ========================================
    # NOUVEAU : Détection d'incohérences
    # ========================================
    import re
    
    incoherence_issues = []
    incoherence_count = 0
    
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().astype(str).head(100)
        
        if len(sample_values) == 0:
            continue
        
        # Détection code postal dans mauvaise colonne
        if 'code' in col_lower and 'postal' in col_lower:
            # Vérifier que ce sont bien des codes postaux (5 chiffres en France)
            non_valid = sample_values[~sample_values.str.match(r'^\d{5}


# ----------------------------------------------------
# OpenAI Reports
# ----------------------------------------------------
def openai_generate_synthesis(df, profil):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        missing = profil['missing_pct'][col]
        schema += f"- {col} ({profil['dtypes'][col]}): {missing:.1f}% manquant, {df[col].nunique()} valeurs uniques\n"

    missing_mean = pd.Series(profil['missing_pct']).mean()
    total_missing = sum(profil['missing_count'].values())
    
    top_missing = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(5)
    top_missing_str = "\n".join([f"  - {col}: {pct:.1f}%" for col, pct in top_missing.items()])
    
    prompt = f"""
    Tu es consultant expert en Data Quality. Analyse ce dataset et fournis un rapport DÉTAILLÉ et STRUCTURÉ.

    ## Données globales
    - Lignes: {profil['rows']:,}
    - Colonnes: {profil['cols']}
    - Score global: {profil['global_score']}%
    - Valeurs manquantes totales: {total_missing:,} ({missing_mean:.2f}% en moyenne)
    - Doublons: {profil['duplicate_rows']}
    - Colonnes constantes: {len(profil['constant_columns'])}
    - Colonnes vides: {len(profil['empty_columns'])}

    ## Top 5 colonnes avec valeurs manquantes
{top_missing_str}

    ## Outliers détectés (IQR)
    {dict(list(profil['outliers'].items())[:5])}

    ## Format attendu:

    ### Synthèse Professionnelle
    Rédige une analyse détaillée (15-20 lignes) couvrant:
    - État général du dataset
    - Problématiques majeures avec impact business
    - Risques pour les analyses
    - Axes d'amélioration prioritaires

    ### Tableau de Priorisation
    Format MARKDOWN:
    
    | Priorité | Problème | Colonnes | Recommandation |
    |----------|----------|----------|----------------|
    | Élevée | [problème] | [colonnes] | [action] |
    
    IMPORTANT:
    - Colonne "Colonnes": MAX 20 caractères
    - Colonne "Recommandation": MAX 60 caractères
    - 5-8 lignes

    ### 5 Quick Wins
    1. **[Titre]**: Description (MAX 100 caractères).
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant senior expert en qualité des données."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


def openai_suggest_tests(df, profil, col_types):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        col_type = col_types.get(col, "unknown")
        missing = profil['missing_pct'][col]
        unique = df[col].nunique()
        schema += f"- {col} ({col_type}): {missing:.1f}% manquant, {unique} uniques\n"

    prompt = f"""
    Expert en data quality testing.
    
    Dataset:
{schema}

    Propose 8-12 tests SPÉCIFIQUES.
    
    Format:
    ## [N]. [Nom]
    **Objectif:** [Pourquoi]
    **Colonnes:** [Colonnes]
    **Méthode:** [Comment]
    **Critère:** [Seuil]
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert en data quality testing."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


# ----------------------------------------------------
# PDF Generation - Fonctions graphiques
# ----------------------------------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_gauge_chart(score):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    if score >= 80:
        color = '#4CAF50'
    elif score >= 60:
        color = '#FFC107'
    else:
        color = '#F44336'
    
    from matplotlib.patches import Wedge
    wedge_bg = Wedge((0.5, 0), 0.4, 0, 180, width=0.15, facecolor='#E0E0E0', edgecolor='white', linewidth=2)
    ax.add_patch(wedge_bg)
    
    angle = score * 1.8
    wedge_score = Wedge((0.5, 0), 0.4, 0, angle, width=0.15, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(wedge_score)
    
    ax.text(0.5, 0.15, f"{score}%", ha='center', va='center', fontsize=32, fontweight='bold', color=color)
    ax.text(0.5, -0.05, "Score Global", ha='center', va='center', fontsize=12, color='#666')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.axis('off')
    
    return fig


def create_missing_chart(profil):
    missing_data = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(10)
    
    if missing_data.empty or missing_data.sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = ['#F44336' if x > 50 else '#FFC107' if x > 20 else '#4CAF50' for x in missing_data.values]
    
    missing_data.plot(kind='barh', ax=ax, color=colors_list, edgecolor='white', linewidth=1.5)
    ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Top 10 des colonnes avec valeurs manquantes", fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(missing_data.values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_types_chart(col_types):
    type_counts = pd.Series(col_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_pie = ['#2196F3', '#4CAF50', '#FFC107', '#F44336', '#9C27B0', '#00BCD4']
    
    wedges, texts, autotexts = ax.pie(
        type_counts.values, 
        labels=type_counts.index, 
        autopct='%1.1f%%',
        colors=colors_pie[:len(type_counts)],
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Distribution des types de colonnes", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def footer(canvas, doc):
    canvas.saveState()
    footer_text = f"Rapport Data Quality - Confidentiel | Page {doc.page}"
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(2*cm, 1.5*cm, footer_text)
    canvas.setStrokeColor(colors.HexColor('#DDDDDD'))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 2*cm, A4[0]-2*cm, 2*cm)
    canvas.restoreState()


# FIN PARTIE 1/2
# La partie 2 contient: build_pdf() et l'interface Streamlit
, na=False)]
            if len(non_valid) > len(sample_values) * 0.2:  # Plus de 20% invalides
                incoherence_count += len(non_valid)
                incoherence_issues.append({
                    'colonne': col,
                    'type': 'Format code postal invalide',
                    'count': len(non_valid),
                    'exemples': non_valid.head(3).tolist()
                })
        
        # Détection email invalide
        elif 'mail' in col_lower or 'email' in col_lower:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}


# ----------------------------------------------------
# OpenAI Reports
# ----------------------------------------------------
def openai_generate_synthesis(df, profil):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        missing = profil['missing_pct'][col]
        schema += f"- {col} ({profil['dtypes'][col]}): {missing:.1f}% manquant, {df[col].nunique()} valeurs uniques\n"

    missing_mean = pd.Series(profil['missing_pct']).mean()
    total_missing = sum(profil['missing_count'].values())
    
    top_missing = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(5)
    top_missing_str = "\n".join([f"  - {col}: {pct:.1f}%" for col, pct in top_missing.items()])
    
    prompt = f"""
    Tu es consultant expert en Data Quality. Analyse ce dataset et fournis un rapport DÉTAILLÉ et STRUCTURÉ.

    ## Données globales
    - Lignes: {profil['rows']:,}
    - Colonnes: {profil['cols']}
    - Score global: {profil['global_score']}%
    - Valeurs manquantes totales: {total_missing:,} ({missing_mean:.2f}% en moyenne)
    - Doublons: {profil['duplicate_rows']}
    - Colonnes constantes: {len(profil['constant_columns'])}
    - Colonnes vides: {len(profil['empty_columns'])}

    ## Top 5 colonnes avec valeurs manquantes
{top_missing_str}

    ## Outliers détectés (IQR)
    {dict(list(profil['outliers'].items())[:5])}

    ## Format attendu:

    ### Synthèse Professionnelle
    Rédige une analyse détaillée (15-20 lignes) couvrant:
    - État général du dataset
    - Problématiques majeures avec impact business
    - Risques pour les analyses
    - Axes d'amélioration prioritaires

    ### Tableau de Priorisation
    Format MARKDOWN:
    
    | Priorité | Problème | Colonnes | Recommandation |
    |----------|----------|----------|----------------|
    | Élevée | [problème] | [colonnes] | [action] |
    
    IMPORTANT:
    - Colonne "Colonnes": MAX 20 caractères
    - Colonne "Recommandation": MAX 60 caractères
    - 5-8 lignes

    ### 5 Quick Wins
    1. **[Titre]**: Description (MAX 100 caractères).
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant senior expert en qualité des données."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


def openai_suggest_tests(df, profil, col_types):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        col_type = col_types.get(col, "unknown")
        missing = profil['missing_pct'][col]
        unique = df[col].nunique()
        schema += f"- {col} ({col_type}): {missing:.1f}% manquant, {unique} uniques\n"

    prompt = f"""
    Expert en data quality testing.
    
    Dataset:
{schema}

    Propose 8-12 tests SPÉCIFIQUES.
    
    Format:
    ## [N]. [Nom]
    **Objectif:** [Pourquoi]
    **Colonnes:** [Colonnes]
    **Méthode:** [Comment]
    **Critère:** [Seuil]
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert en data quality testing."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


# ----------------------------------------------------
# PDF Generation - Fonctions graphiques
# ----------------------------------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_gauge_chart(score):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    if score >= 80:
        color = '#4CAF50'
    elif score >= 60:
        color = '#FFC107'
    else:
        color = '#F44336'
    
    from matplotlib.patches import Wedge
    wedge_bg = Wedge((0.5, 0), 0.4, 0, 180, width=0.15, facecolor='#E0E0E0', edgecolor='white', linewidth=2)
    ax.add_patch(wedge_bg)
    
    angle = score * 1.8
    wedge_score = Wedge((0.5, 0), 0.4, 0, angle, width=0.15, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(wedge_score)
    
    ax.text(0.5, 0.15, f"{score}%", ha='center', va='center', fontsize=32, fontweight='bold', color=color)
    ax.text(0.5, -0.05, "Score Global", ha='center', va='center', fontsize=12, color='#666')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.axis('off')
    
    return fig


def create_missing_chart(profil):
    missing_data = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(10)
    
    if missing_data.empty or missing_data.sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = ['#F44336' if x > 50 else '#FFC107' if x > 20 else '#4CAF50' for x in missing_data.values]
    
    missing_data.plot(kind='barh', ax=ax, color=colors_list, edgecolor='white', linewidth=1.5)
    ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Top 10 des colonnes avec valeurs manquantes", fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(missing_data.values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_types_chart(col_types):
    type_counts = pd.Series(col_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_pie = ['#2196F3', '#4CAF50', '#FFC107', '#F44336', '#9C27B0', '#00BCD4']
    
    wedges, texts, autotexts = ax.pie(
        type_counts.values, 
        labels=type_counts.index, 
        autopct='%1.1f%%',
        colors=colors_pie[:len(type_counts)],
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Distribution des types de colonnes", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def footer(canvas, doc):
    canvas.saveState()
    footer_text = f"Rapport Data Quality - Confidentiel | Page {doc.page}"
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(2*cm, 1.5*cm, footer_text)
    canvas.setStrokeColor(colors.HexColor('#DDDDDD'))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 2*cm, A4[0]-2*cm, 2*cm)
    canvas.restoreState()


# FIN PARTIE 1/2
# La partie 2 contient: build_pdf() et l'interface Streamlit

            non_valid = sample_values[~sample_values.str.match(email_pattern, na=False)]
            if len(non_valid) > len(sample_values) * 0.1:
                incoherence_count += len(non_valid)
                incoherence_issues.append({
                    'colonne': col,
                    'type': 'Format email invalide',
                    'count': len(non_valid),
                    'exemples': non_valid.head(3).tolist()
                })
        
        # Détection téléphone invalide
        elif 'tel' in col_lower or 'phone' in col_lower:
            phone_pattern = r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}


# ----------------------------------------------------
# OpenAI Reports
# ----------------------------------------------------
def openai_generate_synthesis(df, profil):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        missing = profil['missing_pct'][col]
        schema += f"- {col} ({profil['dtypes'][col]}): {missing:.1f}% manquant, {df[col].nunique()} valeurs uniques\n"

    missing_mean = pd.Series(profil['missing_pct']).mean()
    total_missing = sum(profil['missing_count'].values())
    
    top_missing = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(5)
    top_missing_str = "\n".join([f"  - {col}: {pct:.1f}%" for col, pct in top_missing.items()])
    
    prompt = f"""
    Tu es consultant expert en Data Quality. Analyse ce dataset et fournis un rapport DÉTAILLÉ et STRUCTURÉ.

    ## Données globales
    - Lignes: {profil['rows']:,}
    - Colonnes: {profil['cols']}
    - Score global: {profil['global_score']}%
    - Valeurs manquantes totales: {total_missing:,} ({missing_mean:.2f}% en moyenne)
    - Doublons: {profil['duplicate_rows']}
    - Colonnes constantes: {len(profil['constant_columns'])}
    - Colonnes vides: {len(profil['empty_columns'])}

    ## Top 5 colonnes avec valeurs manquantes
{top_missing_str}

    ## Outliers détectés (IQR)
    {dict(list(profil['outliers'].items())[:5])}

    ## Format attendu:

    ### Synthèse Professionnelle
    Rédige une analyse détaillée (15-20 lignes) couvrant:
    - État général du dataset
    - Problématiques majeures avec impact business
    - Risques pour les analyses
    - Axes d'amélioration prioritaires

    ### Tableau de Priorisation
    Format MARKDOWN:
    
    | Priorité | Problème | Colonnes | Recommandation |
    |----------|----------|----------|----------------|
    | Élevée | [problème] | [colonnes] | [action] |
    
    IMPORTANT:
    - Colonne "Colonnes": MAX 20 caractères
    - Colonne "Recommandation": MAX 60 caractères
    - 5-8 lignes

    ### 5 Quick Wins
    1. **[Titre]**: Description (MAX 100 caractères).
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant senior expert en qualité des données."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


def openai_suggest_tests(df, profil, col_types):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        col_type = col_types.get(col, "unknown")
        missing = profil['missing_pct'][col]
        unique = df[col].nunique()
        schema += f"- {col} ({col_type}): {missing:.1f}% manquant, {unique} uniques\n"

    prompt = f"""
    Expert en data quality testing.
    
    Dataset:
{schema}

    Propose 8-12 tests SPÉCIFIQUES.
    
    Format:
    ## [N]. [Nom]
    **Objectif:** [Pourquoi]
    **Colonnes:** [Colonnes]
    **Méthode:** [Comment]
    **Critère:** [Seuil]
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert en data quality testing."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


# ----------------------------------------------------
# PDF Generation - Fonctions graphiques
# ----------------------------------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_gauge_chart(score):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    if score >= 80:
        color = '#4CAF50'
    elif score >= 60:
        color = '#FFC107'
    else:
        color = '#F44336'
    
    from matplotlib.patches import Wedge
    wedge_bg = Wedge((0.5, 0), 0.4, 0, 180, width=0.15, facecolor='#E0E0E0', edgecolor='white', linewidth=2)
    ax.add_patch(wedge_bg)
    
    angle = score * 1.8
    wedge_score = Wedge((0.5, 0), 0.4, 0, angle, width=0.15, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(wedge_score)
    
    ax.text(0.5, 0.15, f"{score}%", ha='center', va='center', fontsize=32, fontweight='bold', color=color)
    ax.text(0.5, -0.05, "Score Global", ha='center', va='center', fontsize=12, color='#666')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.axis('off')
    
    return fig


def create_missing_chart(profil):
    missing_data = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(10)
    
    if missing_data.empty or missing_data.sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = ['#F44336' if x > 50 else '#FFC107' if x > 20 else '#4CAF50' for x in missing_data.values]
    
    missing_data.plot(kind='barh', ax=ax, color=colors_list, edgecolor='white', linewidth=1.5)
    ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Top 10 des colonnes avec valeurs manquantes", fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(missing_data.values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_types_chart(col_types):
    type_counts = pd.Series(col_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_pie = ['#2196F3', '#4CAF50', '#FFC107', '#F44336', '#9C27B0', '#00BCD4']
    
    wedges, texts, autotexts = ax.pie(
        type_counts.values, 
        labels=type_counts.index, 
        autopct='%1.1f%%',
        colors=colors_pie[:len(type_counts)],
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Distribution des types de colonnes", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def footer(canvas, doc):
    canvas.saveState()
    footer_text = f"Rapport Data Quality - Confidentiel | Page {doc.page}"
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(2*cm, 1.5*cm, footer_text)
    canvas.setStrokeColor(colors.HexColor('#DDDDDD'))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 2*cm, A4[0]-2*cm, 2*cm)
    canvas.restoreState()


# FIN PARTIE 1/2
# La partie 2 contient: build_pdf() et l'interface Streamlit

            non_valid = sample_values[~sample_values.str.match(phone_pattern, na=False)]
            if len(non_valid) > len(sample_values) * 0.2:
                incoherence_count += len(non_valid)
                incoherence_issues.append({
                    'colonne': col,
                    'type': 'Format téléphone invalide',
                    'count': len(non_valid),
                    'exemples': non_valid.head(3).tolist()
                })
        
        # Détection date invalide
        elif 'date' in col_lower:
            if df[col].dtype == 'object':
                try:
                    converted = pd.to_datetime(df[col], errors='coerce')
                    non_valid_count = converted.isna().sum() - df[col].isna().sum()
                    if non_valid_count > len(df) * 0.1:
                        incoherence_count += non_valid_count
                        incoherence_issues.append({
                            'colonne': col,
                            'type': 'Format date invalide',
                            'count': int(non_valid_count),
                            'exemples': df[col][converted.isna() & df[col].notna()].head(3).tolist()
                        })
                except:
                    pass
        
        # Détection prénom/nom contenant des chiffres
        elif any(x in col_lower for x in ['nom', 'prenom', 'name', 'firstname', 'lastname']):
            has_digits = sample_values.str.contains(r'\d', na=False)
            if has_digits.sum() > len(sample_values) * 0.05:  # Plus de 5%
                incoherence_count += has_digits.sum()
                incoherence_issues.append({
                    'colonne': col,
                    'type': 'Nom/Prénom contient des chiffres',
                    'count': int(has_digits.sum()),
                    'exemples': sample_values[has_digits].head(3).tolist()
                })
        
        # Détection code (département, commune) contenant du texte
        elif 'code' in col_lower and 'postal' not in col_lower:
            if df[col].dtype == 'object':
                has_letters = sample_values.str.contains(r'[a-zA-Z]', na=False)
                if has_letters.sum() > len(sample_values) * 0.1:
                    incoherence_count += has_letters.sum()
                    incoherence_issues.append({
                        'colonne': col,
                        'type': 'Code contient du texte',
                        'count': int(has_letters.sum()),
                        'exemples': sample_values[has_letters].head(3).tolist()
                    })
    
    profil['incoherence_issues'] = incoherence_issues
    profil['incoherence_count'] = incoherence_count

    # ========================================
    # Score global AMÉLIORÉ avec cohérence
    # ========================================
    missing_mean = pd.Series(profil['missing_pct']).mean() if profil['missing_pct'] else 0
    miss_score = max(0, 100 - missing_mean)
    dup_score = max(0, 100 - (profil["duplicate_rows"]/max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(outliers.values())) if outliers else 0))
    
    # NOUVEAU: Score de cohérence
    coherence_pct = (incoherence_count / max(1, profil["rows"])) * 100
    coherence_score = max(0, 100 - coherence_pct)
    
    # Pondération: 40% complétude, 20% unicité, 15% outliers, 25% cohérence
    profil["global_score"] = round(
        (miss_score * 0.40 + dup_score * 0.20 + out_score * 0.15 + coherence_score * 0.25), 
        1
    )
    profil["coherence_score"] = round(coherence_score, 1)

    return profil


# ----------------------------------------------------
# OpenAI Reports
# ----------------------------------------------------
def openai_generate_synthesis(df, profil):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        missing = profil['missing_pct'][col]
        schema += f"- {col} ({profil['dtypes'][col]}): {missing:.1f}% manquant, {df[col].nunique()} valeurs uniques\n"

    missing_mean = pd.Series(profil['missing_pct']).mean()
    total_missing = sum(profil['missing_count'].values())
    
    top_missing = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(5)
    top_missing_str = "\n".join([f"  - {col}: {pct:.1f}%" for col, pct in top_missing.items()])
    
    prompt = f"""
    Tu es consultant expert en Data Quality. Analyse ce dataset et fournis un rapport DÉTAILLÉ et STRUCTURÉ.

    ## Données globales
    - Lignes: {profil['rows']:,}
    - Colonnes: {profil['cols']}
    - Score global: {profil['global_score']}%
    - Valeurs manquantes totales: {total_missing:,} ({missing_mean:.2f}% en moyenne)
    - Doublons: {profil['duplicate_rows']}
    - Colonnes constantes: {len(profil['constant_columns'])}
    - Colonnes vides: {len(profil['empty_columns'])}

    ## Top 5 colonnes avec valeurs manquantes
{top_missing_str}

    ## Outliers détectés (IQR)
    {dict(list(profil['outliers'].items())[:5])}

    ## Format attendu:

    ### Synthèse Professionnelle
    Rédige une analyse détaillée (15-20 lignes) couvrant:
    - État général du dataset
    - Problématiques majeures avec impact business
    - Risques pour les analyses
    - Axes d'amélioration prioritaires

    ### Tableau de Priorisation
    Format MARKDOWN:
    
    | Priorité | Problème | Colonnes | Recommandation |
    |----------|----------|----------|----------------|
    | Élevée | [problème] | [colonnes] | [action] |
    
    IMPORTANT:
    - Colonne "Colonnes": MAX 20 caractères
    - Colonne "Recommandation": MAX 60 caractères
    - 5-8 lignes

    ### 5 Quick Wins
    1. **[Titre]**: Description (MAX 100 caractères).
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant senior expert en qualité des données."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


def openai_suggest_tests(df, profil, col_types):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns[:15]:
        col_type = col_types.get(col, "unknown")
        missing = profil['missing_pct'][col]
        unique = df[col].nunique()
        schema += f"- {col} ({col_type}): {missing:.1f}% manquant, {unique} uniques\n"

    prompt = f"""
    Expert en data quality testing.
    
    Dataset:
{schema}

    Propose 8-12 tests SPÉCIFIQUES.
    
    Format:
    ## [N]. [Nom]
    **Objectif:** [Pourquoi]
    **Colonnes:** [Colonnes]
    **Méthode:** [Comment]
    **Critère:** [Seuil]
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert en data quality testing."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


# ----------------------------------------------------
# PDF Generation - Fonctions graphiques
# ----------------------------------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def create_gauge_chart(score):
    fig, ax = plt.subplots(figsize=(6, 3))
    
    if score >= 80:
        color = '#4CAF50'
    elif score >= 60:
        color = '#FFC107'
    else:
        color = '#F44336'
    
    from matplotlib.patches import Wedge
    wedge_bg = Wedge((0.5, 0), 0.4, 0, 180, width=0.15, facecolor='#E0E0E0', edgecolor='white', linewidth=2)
    ax.add_patch(wedge_bg)
    
    angle = score * 1.8
    wedge_score = Wedge((0.5, 0), 0.4, 0, angle, width=0.15, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(wedge_score)
    
    ax.text(0.5, 0.15, f"{score}%", ha='center', va='center', fontsize=32, fontweight='bold', color=color)
    ax.text(0.5, -0.05, "Score Global", ha='center', va='center', fontsize=12, color='#666')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.axis('off')
    
    return fig


def create_missing_chart(profil):
    missing_data = pd.Series(profil['missing_pct']).sort_values(ascending=False).head(10)
    
    if missing_data.empty or missing_data.sum() == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = ['#F44336' if x > 50 else '#FFC107' if x > 20 else '#4CAF50' for x in missing_data.values]
    
    missing_data.plot(kind='barh', ax=ax, color=colors_list, edgecolor='white', linewidth=1.5)
    ax.set_xlabel("Pourcentage de valeurs manquantes (%)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Top 10 des colonnes avec valeurs manquantes", fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(missing_data.values):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_types_chart(col_types):
    type_counts = pd.Series(col_types).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_pie = ['#2196F3', '#4CAF50', '#FFC107', '#F44336', '#9C27B0', '#00BCD4']
    
    wedges, texts, autotexts = ax.pie(
        type_counts.values, 
        labels=type_counts.index, 
        autopct='%1.1f%%',
        colors=colors_pie[:len(type_counts)],
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Distribution des types de colonnes", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def footer(canvas, doc):
    canvas.saveState()
    footer_text = f"Rapport Data Quality - Confidentiel | Page {doc.page}"
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(2*cm, 1.5*cm, footer_text)
    canvas.setStrokeColor(colors.HexColor('#DDDDDD'))
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 2*cm, A4[0]-2*cm, 2*cm)
    canvas.restoreState()


# FIN PARTIE 1/2
# La partie 2 contient: build_pdf() et l'interface Streamlit


# streamlit_app.py - PARTIE 2/2
# Coller cette partie APRÈS la partie 1

# ----------------------------------------------------
# PDF Generation - Fonction principale
# ----------------------------------------------------
def build_pdf(report_text, profil, df, col_types, figs_bytes_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, 
        pagesize=A4, 
        rightMargin=2*cm, 
        leftMargin=2*cm, 
        topMargin=2.5*cm, 
        bottomMargin=3*cm
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # PAGE DE GARDE
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=32,
        textColor=colors.HexColor('#1976D2'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=16,
        textColor=colors.HexColor('#666666'),
        spaceAfter=50,
        alignment=TA_CENTER
    )
    
    story.append(Spacer(1, 5*cm))
    story.append(Paragraph("Rapport d'Audit", title_style))
    story.append(Paragraph("Data Quality", title_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", subtitle_style))
    
    meta_data = [
        ['', ''],
        ['📊 Dataset', f"{profil['rows']:,} lignes × {profil['cols']} colonnes"],
        ['✓ Score Global', f"{profil['global_score']}%"],
        ['⚠ Valeurs Manquantes', f"{sum(profil['missing_count'].values()):,}"],
        ['🔄 Doublons', f"{profil['duplicate_rows']:,}"],
        ['', '']
    ]
    
    meta_table = Table(meta_data, colWidths=[5*cm, 10*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,1), (-1,-2), colors.HexColor('#F5F5F5')),
        ('TEXTCOLOR', (0,1), (-1,-2), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,1), (0,-2), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,-2), 12),
        ('GRID', (0,1), (-1,-2), 1, colors.HexColor('#DDDDDD')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#FAFAFA')]),
        ('TOPPADDING', (0,1), (-1,-2), 12),
        ('BOTTOMPADDING', (0,1), (-1,-2), 12),
    ]))
    
    story.append(meta_table)
    story.append(PageBreak())
    
    # RÉSUMÉ EXÉCUTIF
    story.append(Paragraph("Résumé Exécutif", styles['Heading1']))
    story.append(Spacer(1, 0.3*cm))
    
    gauge_fig = create_gauge_chart(profil['global_score'])
    gauge_bytes = fig_to_bytes(gauge_fig)
    gauge_img = Image.open(gauge_bytes)
    
    gauge_buf = io.BytesIO()
    gauge_img.save(gauge_buf, format="PNG")
    gauge_buf.seek(0)
    
    story.append(RLImage(gauge_buf, width=12*cm, height=6*cm))
    story.append(Spacer(1, 0.5*cm))
    
    kpi_data = [
        ['Indicateur', 'Valeur', 'État'],
        ['Complétude', f"{100 - pd.Series(profil['missing_pct']).mean():.1f}%", 
         '✓' if pd.Series(profil['missing_pct']).mean() < 10 else '⚠'],
        ['Unicité', f"{100 - (profil['duplicate_rows']/profil['rows']*100):.1f}%",
         '✓' if profil['duplicate_rows'] == 0 else '✗'],
        ['Outliers', f"{sum(profil['outliers'].values())} détectés",
         '✓' if sum(profil['outliers'].values()) < 50 else '⚠'],
        ['Colonnes constantes', f"{len(profil['constant_columns'])}",
         '✓' if len(profil['constant_columns']) == 0 else '⚠']
    ]
    
    kpi_table = Table(kpi_data, colWidths=[6*cm, 5*cm, 3*cm])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9F9F9')]),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    
    story.append(kpi_table)
    story.append(PageBreak())
    
    # SYNTHÈSE DÉTAILLÉE
    story.append(Paragraph("Analyse Détaillée", styles['Heading1']))
    story.append(Spacer(1, 0.5*cm))
    
    sections = report_text.split("###")
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split("\n")
        section_title = lines[0].strip()
        section_content = "\n".join(lines[1:]).strip()
        
        if section_title:
            clean_title = escape_for_reportlab(section_title)
            story.append(Paragraph(clean_title, styles['Heading2']))
            story.append(Spacer(1, 0.3*cm))
        
        if "|" in section_content and "---" in section_content:
            table_lines = [line for line in section_content.split("\n") if line.strip().startswith("|")]
            
            if len(table_lines) > 2:
                table_data = []
                for line in table_lines:
                    cells = [cell.strip() for cell in line.split("|")[1:-1]]
                    if "---" not in line:
                        clean_cells = []
                        for i, cell in enumerate(cells):
                            cleaned = escape_for_reportlab(cell)
                            if i == 2 and len(cleaned) > 35:
                                cleaned = cleaned[:32] + "..."
                            elif i == 3 and len(cleaned) > 80:
                                cleaned = cleaned[:77] + "..."
                            clean_cells.append(cleaned)
                        table_data.append(clean_cells)
                
                if table_data:
                    num_cols = len(table_data[0])
                    if num_cols == 4:
                        col_widths = [2.5*cm, 4.5*cm, 3.5*cm, 6.5*cm]
                    else:
                        col_widths = [17*cm / num_cols] * num_cols
                    
                    pdf_table = Table(table_data, colWidths=col_widths)
                    
                    table_style = [
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1976D2')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,-1), 8),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                        ('TOPPADDING', (0,0), (-1,-1), 6),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ]
                    
                    for i, row in enumerate(table_data[1:], start=1):
                        if 'Élevée' in row[0] or 'Haute' in row[0]:
                            table_style.append(('BACKGROUND', (0,i), (0,i), colors.HexColor('#FFCDD2')))
                        elif 'Moyenne' in row[0]:
                            table_style.append(('BACKGROUND', (0,i), (0,i), colors.HexColor('#FFF9C4')))
                        elif 'Faible' in row[0] or 'Basse' in row[0]:
                            table_style.append(('BACKGROUND', (0,i), (0,i), colors.HexColor('#C8E6C9')))
                    
                    pdf_table.setStyle(TableStyle(table_style))
                    story.append(pdf_table)
        else:
            paragraphs = section_content.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                try:
                    clean_para = escape_for_reportlab(para)
                    
                    if len(clean_para) > 0 and clean_para[0].isdigit() and ". " in clean_para[:5]:
                        story.append(Paragraph(clean_para, styles['Normal']))
                    else:
                        story.append(Paragraph(clean_para, styles['BodyText']))
                    
                    story.append(Spacer(1, 0.2*cm))
                except Exception:
                    simple_text = para.replace('<', '').replace('>', '').replace('&', '')
                    story.append(Paragraph(simple_text, styles['Normal']))
                    story.append(Spacer(1, 0.2*cm))
        
        story.append(Spacer(1, 0.3*cm))
    
    story.append(PageBreak())
    
    # VISUALISATIONS
    story.append(Paragraph("Visualisations", styles['Heading1']))
    story.append(Spacer(1, 0.5*cm))
    
    missing_fig = create_missing_chart(profil)
    if missing_fig:
        story.append(Paragraph("Colonnes avec valeurs manquantes", styles['Heading2']))
        missing_bytes = fig_to_bytes(missing_fig)
        missing_img = Image.open(missing_bytes)
        
        max_w = A4[0] - 4*cm
        ratio = max_w / missing_img.size[0]
        
        missing_buf = io.BytesIO()
        missing_img.save(missing_buf, format="PNG")
        missing_buf.seek(0)
        
        story.append(RLImage(missing_buf, width=max_w, height=missing_img.size[1]*ratio))
        story.append(Spacer(1, 1*cm))
    
    types_fig = create_types_chart(col_types)
    story.append(Paragraph("Types de colonnes détectés", styles['Heading2']))
    types_bytes = fig_to_bytes(types_fig)
    types_img = Image.open(types_bytes)
    
    max_w = A4[0] - 4*cm
    ratio = max_w / types_img.size[0]
    
    types_buf = io.BytesIO()
    types_img.save(types_buf, format="PNG")
    types_buf.seek(0)
    
    story.append(RLImage(types_buf, width=max_w, height=types_img.size[1]*ratio))
    story.append(Spacer(1, 1*cm))
    
    for idx, figb in enumerate(figs_bytes_list):
        story.append(Paragraph(f"Heatmap des outliers", styles['Heading2']))
        img = Image.open(figb)
        
        max_w = A4[0] - 4*cm
        ratio = max_w / img.size[0]
        
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)
        
        story.append(RLImage(img_buf, width=max_w, height=img.size[1]*ratio))
        story.append(Spacer(1, 0.5*cm))
    
    # PLAN D'ACTION
    story.append(PageBreak())
    story.append(Paragraph("Plan d'Action Recommandé", styles['Heading1']))
    story.append(Spacer(1, 0.5*cm))
    
    action_plan = [
        ['Phase', 'Actions', 'Timeline', 'Priorité'],
        ['1. Audit', 'Documenter état actuel\nIdentifier parties prenantes', '1 sem', 'Élevée'],
        ['2. Nettoyage', 'Traiter valeurs manquantes\nSupprimer doublons', '2 sem', 'Élevée'],
        ['3. Standard', 'Harmoniser formats\nValider outliers', '3 sem', 'Moyenne'],
        ['4. Contrôles', 'Règles validation\nMonitoring auto', '4 sem', 'Moyenne'],
        ['5. Doc', 'Guide utilisateur\nDictionnaire', '1 sem', 'Basse'],
    ]
    
    action_table = Table(action_plan, colWidths=[2.5*cm, 6.5*cm, 2.5*cm, 2.5*cm])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9F9F9')]),
    ]))
    
    story.append(action_table)
    story.append(Spacer(1, 1*cm))
    
    story.append(Paragraph("Ressources Nécessaires", styles['Heading2']))
    story.append(Spacer(1, 0.3*cm))
    
    resources_text = """
    • <b>Équipe :</b> 1 Data Analyst + 1 Data Engineer (temps partiel)<br/>
    • <b>Outils :</b> Outils de data quality existants<br/>
    • <b>Budget :</b> Formation et automatisations<br/>
    • <b>Durée estimée :</b> 8-12 semaines
    """
    story.append(Paragraph(resources_text, styles['BodyText']))
    
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buf.seek(0)
    return buf


# ----------------------------------------------------
# INTERFACE STREAMLIT
# ----------------------------------------------------
POWERBI_CSS = """
<style>
section.main > div.block-container { max-width: 1400px; }
h1 { font-family: "Segoe UI", sans-serif; }
div.block-container { padding-top: 18px; padding-left:18px; padding-right:18px; }
</style>
"""

st.set_page_config(page_title="Data Quality App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Testez la qualité de vos données", "Contact"])


if page == "Testez la qualité de vos données":

    st.markdown(POWERBI_CSS, unsafe_allow_html=True)
    st.title("📊 Data Quality Dashboard")

    if OPENAI_API_KEY is None:
        st.warning("⚠️ OpenAI non configuré.")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        df = load_dataframe(uploaded_file.name, file_bytes)

        if df is None:
            st.error("❌ Impossible de lire le fichier.")
        else:
            df_dict = df.to_dict('list')
            profil = profile_data_quality(df_dict)
            col_types = detect_column_types(df_dict)

            # KPI TILES
            tile_style = """
            <div style="background-color: {bg}; padding: 14px 18px; border-radius: 10px;
                color: {text}; box-shadow: 0 6px 18px rgba(0,0,0,0.08);
                display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size:14px; color:{muted};">{label}</div>
                    <div style="font-size:26px; font-weight:700; margin-top:6px;">{value}</div>
                    <div style="font-size:12px; opacity:0.9; margin-top:4px;">{subtitle}</div>
                </div>
                <div style="font-size:36px;">{icon}</div>
            </div>
            """

            c1, c2, c3, c4 = st.columns([1,1,1,1], gap="large")

            c1.markdown(tile_style.format(
                bg="#F2C811", text="#2B2B2B", muted="#563C00",
                label="Score global", value=f"{profil['global_score']}%",
                subtitle="Indice synthétique", icon="📊"), unsafe_allow_html=True)

            c2.markdown(tile_style.format(
                bg="white", text="#2B2B2B", muted="#666",
                label="Valeurs manquantes", value=int(sum(profil["missing_count"].values())),
                subtitle="Total de NA", icon="❗"), unsafe_allow_html=True)

            c3.markdown(tile_style.format(
                bg="#118DFF", text="#FFFFFF", muted="#DDF4FF",
                label="Doublons", value=profil["duplicate_rows"],
                subtitle="Lignes dupliquées", icon="📑"), unsafe_allow_html=True)

            c4.markdown(tile_style.format(
                bg="#f6f6f6", text="#2B2B2B", muted="#666",
                label="Colonnes vides/constantes",
                value=len(profil["empty_columns"]) + len(profil["constant_columns"]),
                subtitle="Colonnes sans variance", icon="📦"), unsafe_allow_html=True)

            st.markdown("---")
            
            # ========================================
            # NOUVEAU : Affichage des incohérences
            # ========================================
            if profil.get('incoherence_count', 0) > 0:
                st.warning(f"⚠️ **{profil['incoherence_count']} incohérences détectées** (Score cohérence : {profil.get('coherence_score', 100)}%)")
                
                with st.expander("🔍 Détails des incohérences", expanded=True):
                    for issue in profil['incoherence_issues']:
                        st.error(f"**{issue['colonne']}** : {issue['type']} ({issue['count']} valeurs)")
                        st.write(f"Exemples : {', '.join(map(str, issue['exemples']))}")
                        st.write("---")
                
                st.info("💡 Ces incohérences réduisent votre score de qualité. Corrigez-les pour améliorer la fiabilité de vos données.")

            st.markdown("---")

            # TYPOLOGIE
            st.subheader("🏷️ Typologie des colonnes")
            
            type_counts = pd.Series(col_types).value_counts()
            col_type1, col_type2 = st.columns([1, 2])
            
            with col_type1:
                st.dataframe(type_counts.to_frame('Nombre'), use_container_width=True)
            
            with col_type2:
                fig_types, ax_types = plt.subplots(figsize=(8, 4))
                type_counts.plot(kind='barh', ax=ax_types, color='#118DFF')
                ax_types.set_title("Distribution des types de colonnes")
                ax_types.set_xlabel("Nombre de colonnes")
                st.pyplot(fig_types)

            st.markdown("---")
            st.subheader("📄 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # OUTLIERS
            st.markdown("---")
            st.subheader("🔥 Heatmap des Outliers")

            if profil["outliers"] and sum(profil["outliers"].values()) > 0:
                outlier_df = pd.DataFrame(
                    profil["outliers"].values(),
                    index=profil["outliers"].keys(),
                    columns=["outliers"]
                ).sort_values("outliers", ascending=False)

                fig1, ax1 = plt.subplots(figsize=(8, max(2, len(outlier_df)*0.35)))

                sns.heatmap(outlier_df, annot=True, fmt="d",
                            cmap=sns.color_palette("Reds", as_cmap=True),
                            linewidths=0.6, linecolor="white",
                            cbar_kws={'label': 'Outliers'},
                            ax=ax1)

                ax1.set_title("Outliers détectés (IQR)", fontsize=13)
                ax1.set_ylabel("")
                ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

                st.pyplot(fig1)
            else:
                st.info("✅ Aucun outlier détecté ou aucune colonne numérique dans le dataset.")
                fig1 = None

            # SYNTHÈSE
            st.markdown("---")
            st.subheader("🧠 Synthèse globale")

            with st.spinner("Analyse IA..."):
                synthesis = openai_generate_synthesis(df, profil)

            st.markdown(synthesis)

            # TESTS
            st.markdown("---")
            st.subheader("🧪 Tests suggérés")
            
            with st.spinner("Génération..."):
                tests = openai_suggest_tests(df, profil, col_types)
            
            st.markdown(tests)

            # Suite dans le prochain message...
            # (Profiling colonnes, suggestions, PDF export)

            # PROFILING COLONNES
            st.markdown("---")
            st.subheader("📊 Profiling détaillé")
            
            col_select = st.selectbox("Colonne", df.columns)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Type", col_types.get(col_select, "unknown"))
                st.metric("Pandas", str(df[col_select].dtype))
            
            with col2:
                st.metric("Manquantes", f"{(df[col_select].isna().mean()*100):.1f}%")
                st.metric("Uniques", df[col_select].nunique())
            
            with col3:
                st.metric("Remplissage", f"{((1-df[col_select].isna().mean())*100):.1f}%")
                st.metric("Cardinalité", f"{(df[col_select].nunique()/len(df)*100):.1f}%")
            
            if pd.api.types.is_numeric_dtype(df[col_select]):
                fig_prof, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                df[col_select].dropna().hist(bins=30, ax=ax1, color='#118DFF')
                ax1.set_title(f"Distribution")
                ax1.grid(True, alpha=0.3)
                
                df[col_select].dropna().plot(kind='box', ax=ax2, color='#118DFF')
                ax2.set_title(f"Boxplot")
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig_prof)
                st.dataframe(df[col_select].describe().to_frame())
            else:
                top_values = df[col_select].value_counts().head(10)
                fig_prof, ax = plt.subplots(figsize=(10, 5))
                top_values.plot(kind='barh', ax=ax, color='#118DFF')
                ax.set_title(f"Top 10")
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig_prof)
                st.dataframe(df[col_select].value_counts().head(20))

            # SUGGESTIONS
            st.markdown("---")
            st.subheader("🎯 Suggestions")
            
            suggestions = []
            
            if profil['empty_columns']:
                suggestions.append({
                    'action': 'Supprimer vides',
                    'colonnes': ', '.join(profil['empty_columns'][:3]),
                    'impact': f"{len(profil['empty_columns'])} col",
                    'code': f"df.drop(columns={profil['empty_columns']})"
                })
            
            if profil['duplicate_rows'] > 0:
                suggestions.append({
                    'action': 'Supprimer doublons',
                    'colonnes': 'Toutes',
                    'impact': f"{profil['duplicate_rows']} lignes",
                    'code': "df.drop_duplicates()"
                })
            
            missing_pct_series = pd.Series(profil['missing_pct'])
            high_missing = missing_pct_series[missing_pct_series > 0].sort_values(ascending=False)
            
            for col in high_missing.head(5).index:
                missing_pct = high_missing[col]
                suggestions.append({
                    'action': f'Imputer {col}',
                    'colonnes': col,
                    'impact': f"{missing_pct:.1f}%",
                    'code': f"df['{col}'].fillna(df['{col}'].median())"
                })
            
            if suggestions:
                sugg_df = pd.DataFrame(suggestions)
                st.dataframe(sugg_df, use_container_width=True)
                
                with st.expander("📝 Code"):
                    code = "import pandas as pd\n\n"
                    for s in suggestions:
                        code += f"# {s['action']}\n{s['code']}\n\n"
                    st.code(code, language='python')
            else:
                st.success("✅ Données OK!")

            # PDF EXPORT
            st.markdown("---")
            st.subheader("📄 Export PDF")
            
            if fig1 is not None:
                fig_bytes = [fig_to_bytes(fig1)]
            else:
                fig_bytes = []

            with st.spinner("PDF..."):
                pdf_buffer = build_pdf(synthesis, profil, df, col_types, fig_bytes)

            st.download_button(
                label="📄 Télécharger PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary"
            )


elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**Email :** soumanoseydou@icloud.com")
    st.write("**Tel :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** linkedin.com/in/seydou-soumano")
    st.write("**GitHub :** github.com/Ssoumano")
