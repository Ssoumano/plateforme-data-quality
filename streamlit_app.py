# streamlit_app.py
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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from PIL import Image
import base64


# ----------------------------------------------------
# 🔧 FIX : Nettoyage du texte OpenAI
# ----------------------------------------------------
def clean_ai_text(text: str) -> str:
    text = text.replace("`", "")
    text = text.replace("* *", " ")
    text = text.replace("** **", " ")
    text = text.replace("• •", "• ")
    text = text.replace("\u200b", "")  # caractères invisibles
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
        # Ignorer les colonnes vides
        if df[col].isna().all():
            types[col] = 'empty'
            continue
        
        # Tenter de convertir en datetime
        try:
            if df[col].dtype == 'object':
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.8:
                    types[col] = 'datetime'
                    continue
        except:
            pass
        
        # Vérifier si numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  # Peu de valeurs uniques
                types[col] = 'categorical'
            else:
                types[col] = 'numeric'
        else:
            # Vérifier si binaire
            nunique = df[col].nunique()
            if nunique == 2:
                types[col] = 'binary'
            elif nunique / len(df) < 0.05:
                types[col] = 'categorical'
            else:
                types[col] = 'text'
    
    return types


# ----------------------------------------------------
# Data Profiling
# ----------------------------------------------------
@st.cache_data
def profile_data_quality(df_dict) -> dict:
    """Profile data quality with caching"""
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
    profil['numeric_stats'] = numeric.describe().T.to_dict()

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

    # Global score
    missing_mean = pd.Series(profil['missing_pct']).mean()
    miss_score = max(0, 100 - missing_mean)
    dup_score = max(0, 100 - (profil["duplicate_rows"]/max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(outliers.values())) if outliers else 0))
    profil["global_score"] = round((miss_score*0.5 + dup_score*0.3 + out_score*0.2), 1)

    return profil


# ----------------------------------------------------
# OpenAI Reports (Synthèse + Tests)
# ----------------------------------------------------
def openai_generate_synthesis(df, profil):
    if client is None:
        return "OpenAI non configuré. Ajoute OPENAI_API_KEY dans les secrets Streamlit."

    schema = ""
    for col in df.columns[:15]:  # Limiter à 15 colonnes pour le prompt
        missing = profil['missing_pct'][col]
        schema += f"- {col} ({profil['dtypes'][col]}): {missing:.1f}% manquant, {df[col].nunique()} valeurs uniques\n"

    missing_mean = pd.Series(profil['missing_pct']).mean()
    total_missing = sum(profil['missing_count'].values())
    
    # Top colonnes avec valeurs manquantes
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

    ## Schéma (extrait)
{schema}

    ## Format attendu de ta réponse:

    ### Synthèse Professionnelle
    Rédige une analyse détaillée (15-20 lignes) couvrant:
    - État général du dataset (points forts et faiblesses)
    - Problématiques majeures identifiées avec impact business
    - Risques pour les analyses (biais, fiabilité, cohérence)
    - Axes d'amélioration prioritaires

    ### Tableau de Priorisation
    Format MARKDOWN uniquement:
    
    | Priorité | Problème | Colonnes concernées | Impact | Recommandation |
    |----------|----------|---------------------|--------|----------------|
    | Élevée | [problème détaillé] | [colonnes] | [impact chiffré] | [action concrète] |
    
    Minimum 5 lignes, maximum 8 lignes.

    ### 5 Quick Wins
    Liste numérotée d'actions rapides et concrètes, format:
    1. **[Titre de l'action]** : Description précise de l'action et bénéfice attendu (1-2 phrases).

    IMPORTANT: 
    - Sois PRÉCIS avec les noms de colonnes exacts
    - Quantifie les impacts (nombre de lignes, % de données)
    - Priorise selon criticité réelle
    - Évite les généralités, sois actionnable
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant senior expert en qualité des données. Tu fournis des analyses détaillées, structurées et actionnables."},
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
        schema += f"- {col} ({col_type}, pandas: {profil['dtypes'][col]}): {missing:.1f}% manquant, {unique} valeurs uniques\n"

    prompt = f"""
    Tu es expert en data quality testing.
    
    Voici le schéma d'un dataset:
{schema}

    Dataset: {profil['rows']} lignes, {profil['cols']} colonnes
    
    Propose 8-12 tests de qualité SPÉCIFIQUES et ACTIONNABLES pour ce dataset.
    
    Format pour chaque test:
    
    ## [Numéro]. [Nom du test]
    **Objectif:** [Pourquoi ce test?]
    **Colonnes:** [Colonnes exactes à tester]
    **Méthode:** [Comment tester concrètement]
    **Critère de succès:** [Seuil de validation]
    
    Exemple:
    ## 1. Unicité des Identifiants
    **Objectif:** Vérifier qu'il n'y a pas de doublons dans les clés primaires
    **Colonnes:** code_departement, code_commune, numero
    **Méthode:** Vérifier que la combinaison (code_departement, code_commune, numero) est unique
    **Critère de succès:** 0 doublon détecté
    
    Couvre ces dimensions:
    - Complétude (champs obligatoires non nuls)
    - Validité (formats, types, plages de valeurs)
    - Cohérence (relations entre colonnes, règles métier)
    - Unicité (identifiants, clés)
    - Exactitude (valeurs aberrantes, outliers)
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un expert en data quality testing. Tu proposes des tests précis, mesurables et actionnables."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


# ----------------------------------------------------
# PDF Generation
# ----------------------------------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(report_text, profil, df, figs_bytes_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Rapport Data Quality", styles["Title"]))
    story.append(Spacer(1, 12))

    # Metadata table
    meta = [
        ["Lignes", profil['rows']],
        ["Colonnes", profil['cols']],
        ["Score global", f"{profil['global_score']}%"],
        ["Valeurs manquantes (total)", int(sum(profil['missing_count'].values()))],
        ["Doublons", profil['duplicate_rows']],
    ]
    t = Table(meta, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Synthèse AI
    story.append(Paragraph("Synthèse", styles["Heading2"]))
    for para in report_text.split("\n\n"):
        story.append(Paragraph(para.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # Figures
    for idx, figb in enumerate(figs_bytes_list):
        story.append(Paragraph(f"Figure {idx+1}", styles["Heading2"]))
        img = Image.open(figb)

        max_w = A4[0] - 4*cm
        ratio = max_w / img.size[0]
        new_w = max_w
        new_h = img.size[1] * ratio

        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)

        story.append(RLImage(img_buf, width=new_w, height=new_h))
        story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)
    return buf


# ----------------------------------------------------
# THEME Power BI
# ----------------------------------------------------
POWERBI_CSS = """
<style>
section.main > div.block-container { max-width: 1400px; }
h1 { font-family: "Segoe UI", sans-serif; }

.kpi-val { font-size: 28px; font-weight:700; color:#111; }
.kpi-label { font-size: 12px; color:#222; }

div.block-container { padding-top: 18px; padding-left:18px; padding-right:18px; }
</style>
"""

st.set_page_config(page_title="Data Quality App", layout="wide")


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Testez la qualité de vos données", "Contact"])


# ----------------------------------------------------
# PAGE : Data Quality
# ----------------------------------------------------
if page == "Testez la qualité de vos données":

    st.markdown(POWERBI_CSS, unsafe_allow_html=True)
    st.title("📊 Data Quality Dashboard ")

    if OPENAI_API_KEY is None:
        st.warning("⚠️ OpenAI non configuré. Ajoute OPENAI_API_KEY dans les secrets Streamlit.")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Load avec cache
        file_bytes = uploaded_file.getvalue()
        df = load_dataframe(uploaded_file.name, file_bytes)

        if df is None:
            st.error("❌ Impossible de lire le fichier.")
        else:
            # Convertir en dict pour le cache
            df_dict = df.to_dict('list')
            profil = profile_data_quality(df_dict)
            col_types = detect_column_types(df_dict)

            # ----------------------------------------------------
            # KPI Tiles
            # ----------------------------------------------------
            tile_style = """
            <div style="
                background-color: {bg};
                padding: 14px 18px;
                border-radius: 10px;
                color: {text};
                width: 100%;
                box-shadow: 0 6px 18px rgba(0,0,0,0.08);
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <div>
                    <div style="font-size:14px; color:{muted};">{label}</div>
                    <div style="font-size:26px; font-weight:700; margin-top:6px;">{value}</div>
                    <div style="font-size:12px; opacity:0.9; margin-top:4px;">{subtitle}</div>
                </div>
                <div style="font-size:36px;">{icon}</div>
            </div>
            """

            yellow = "#F2C811"
            blue = "#118DFF"
            dark = "#2B2B2B"
            white = "#FFFFFF"

            c1, c2, c3, c4 = st.columns([1,1,1,1], gap="large")

            c1.markdown(tile_style.format(
                bg=yellow, text=dark, muted="#563C00",
                label="Score global", value=f"{profil['global_score']}%",
                subtitle="Indice synthétique", icon="📊"), unsafe_allow_html=True)

            c2.markdown(tile_style.format(
                bg="white", text=dark, muted="#666",
                label="Valeurs manquantes", value=int(sum(profil["missing_count"].values())),
                subtitle="Total de NA", icon="❗"), unsafe_allow_html=True)

            c3.markdown(tile_style.format(
                bg=blue, text=white, muted="#DDF4FF",
                label="Doublons", value=profil["duplicate_rows"],
                subtitle="Lignes dupliquées", icon="📑"), unsafe_allow_html=True)

            c4.markdown(tile_style.format(
                bg="#f6f6f6", text=dark, muted="#666",
                label="Colonnes vides/constantes",
                value=len(profil["empty_columns"]) + len(profil["constant_columns"]),
                subtitle="Colonnes sans variance", icon="📦"), unsafe_allow_html=True)

            st.markdown("---")

            # ----------------------------------------------------
            # Récapitulatif des types de colonnes
            # ----------------------------------------------------
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

            # ----------------------------------------------------
            # DataFrame Preview
            # ----------------------------------------------------
            st.subheader("📄 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # ----------------------------------------------------
            # Outlier Heatmap
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("🔥 Heatmap des Outliers (IQR)")

            outlier_df = pd.DataFrame(
                profil["outliers"].values(),
                index=profil["outliers"].keys(),
                columns=["outliers"]
            ).sort_values("outliers", ascending=False)

            fig1, ax1 = plt.subplots(figsize=(8, max(2, len(outlier_df)*0.35)))

            sns.heatmap(outlier_df, annot=True, fmt="d",
                        cmap=sns.color_palette("Reds", as_cmap=True),
                        linewidths=0.6, linecolor="white",
                        cbar_kws={'label': 'Nombre d\'outliers'},
                        ax=ax1)

            ax1.set_title("Outliers détectés par colonne (IQR)", fontsize=13)
            ax1.set_ylabel("")
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

            st.pyplot(fig1)

            # ----------------------------------------------------
            # Synthèse OpenAI
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("🧠 Synthèse globale")

            with st.spinner("Analyse IA en cours..."):
                synthesis = openai_generate_synthesis(df, profil)

            st.markdown(synthesis)

            # ----------------------------------------------------
            # Tests OpenAI
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("🧪 Tests complémentaires suggérés")
            
            with st.spinner("Génération des tests recommandés..."):
                tests = openai_suggest_tests(df, profil, col_types)
            
            st.markdown(tests)

            # ----------------------------------------------------
            # 📊 Profiling détaillé des colonnes
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("📊 Profiling détaillé des colonnes")
            
            col_select = st.selectbox("Sélectionnez une colonne à analyser", df.columns)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Type détecté", col_types.get(col_select, "unknown"))
                st.metric("Type pandas", str(df[col_select].dtype))
                st.metric("Valeurs uniques", df[col_select].nunique())
            
            with col2:
                st.metric("Valeurs manquantes", f"{(df[col_select].isna().mean()*100):.1f}%")
                st.metric("Valeurs nulles", df[col_select].isna().sum())
            
            with col3:
                st.metric("Taux de remplissage", f"{((1-df[col_select].isna().mean())*100):.1f}%")
                st.metric("Cardinalité", f"{(df[col_select].nunique()/len(df)*100):.1f}%")
            
            # Visualisation selon le type
            if pd.api.types.is_numeric_dtype(df[col_select]):
                fig_prof, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histogramme
                df[col_select].dropna().hist(bins=30, ax=ax1, color='#118DFF', edgecolor='white')
                ax1.set_title(f"Distribution de {col_select}")
                ax1.set_xlabel("Valeur")
                ax1.set_ylabel("Fréquence")
                ax1.grid(True, alpha=0.3)
                
                # Boxplot
                df[col_select].dropna().plot(kind='box', ax=ax2, color='#118DFF')
                ax2.set_title(f"Boxplot de {col_select}")
                ax2.set_ylabel("Valeur")
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig_prof)
                
                # Statistiques descriptives
                st.write("**Statistiques descriptives:**")
                st.dataframe(df[col_select].describe().to_frame())
                
            else:
                # Pour colonnes catégorielles
                top_values = df[col_select].value_counts().head(10)
                
                fig_prof, ax = plt.subplots(figsize=(10, 6))
                top_values.plot(kind='barh', ax=ax, color='#118DFF')
                ax.set_title(f"Top 10 des valeurs - {col_select}")
                ax.set_xlabel("Fréquence")
                ax.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig_prof)
                
                st.write("**Top 20 des valeurs:**")
                st.dataframe(df[col_select].value_counts().head(20).to_frame())

            # ----------------------------------------------------
            # 🔍 Matrice de corrélation des valeurs manquantes
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("🔍 Matrice de corrélation des valeurs manquantes")
            
            missing_matrix = df.isna().astype(int)
            
            if missing_matrix.sum().sum() > 0:
                missing_corr = missing_matrix.corr()
                
                fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
                sns.heatmap(missing_corr, annot=True, fmt=".2f", 
                           cmap='coolwarm', center=0,
                           square=True, linewidths=0.5,
                           cbar_kws={"shrink": 0.8},
                           ax=ax_corr)
                ax_corr.set_title("Corrélation entre colonnes avec valeurs manquantes\n(1 = manquent toujours ensemble)")
                
                st.pyplot(fig_corr)
                
                st.info("💡 Une corrélation élevée (>0.7) indique que les valeurs manquent souvent ensemble dans ces colonnes.")
            else:
                st.success("✅ Aucune valeur manquante détectée!")

            # ----------------------------------------------------
            # 📈 Évolution temporelle (si colonne date détectée)
            # ----------------------------------------------------
            date_cols = [col for col in df.columns if col_types.get(col) == 'datetime' 
                        or 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                st.markdown("---")
                st.subheader("📈 Analyse temporelle")
                
                date_col = st.selectbox("Sélectionnez une colonne de date", date_cols)
                
                try:
                    df_temp = df.copy()
                    if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
                        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                    
                    df_temp = df_temp.sort_values(date_col)
                    df_temp['missing_count'] = df_temp.isna().sum(axis=1)
                    
                    fig_time, ax_time = plt.subplots(figsize=(12, 5))
                    df_temp.groupby(df_temp[date_col].dt.to_period('M'))['missing_count'].mean().plot(
                        ax=ax_time, color='#F2C811', marker='o', linewidth=2
                    )
                    ax_time.set_title("Évolution des valeurs manquantes par mois")
                    ax_time.set_xlabel("Période")
                    ax_time.set_ylabel("Moyenne de valeurs manquantes par ligne")
                    ax_time.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_time)
                except Exception as e:
                    st.warning(f"Impossible d'analyser la dimension temporelle: {str(e)}")

            # ----------------------------------------------------
            # 🎯 Suggestions de nettoyage automatiques
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("🎯 Suggestions de nettoyage")
            
            suggestions = []
            
            # Suggestion 1: Supprimer colonnes vides
            if profil['empty_columns']:
                suggestions.append({
                    'action': 'Supprimer colonnes vides',
                    'colonnes': ', '.join(profil['empty_columns']),
                    'impact': f"{len(profil['empty_columns'])} colonnes",
                    'code': f"df = df.drop(columns={profil['empty_columns']})"
                })
            
            # Suggestion 2: Supprimer colonnes constantes
            if profil['constant_columns']:
                suggestions.append({
                    'action': 'Supprimer colonnes constantes',
                    'colonnes': ', '.join(profil['constant_columns']),
                    'impact': f"{len(profil['constant_columns'])} colonnes",
                    'code': f"df = df.drop(columns={profil['constant_columns']})"
                })
            
            # Suggestion 3: Traiter les doublons
            if profil['duplicate_rows'] > 0:
                suggestions.append({
                    'action': 'Supprimer les doublons',
                    'colonnes': 'Toutes',
                    'impact': f"{profil['duplicate_rows']} lignes",
                    'code': "df = df.drop_duplicates()"
                })
            
            # Suggestion 4: Imputer les valeurs manquantes
            missing_pct_series = pd.Series(profil['missing_pct'])
            high_missing = missing_pct_series[missing_pct_series > 0].sort_values(ascending=False)
            
            if len(high_missing) > 0:
                # Ajouter suggestion pour chaque colonne avec des valeurs manquantes
                for col in high_missing.index:
                    missing_pct = high_missing[col]
                    missing_count = profil['missing_count'][col]
                    
                    if pd.api.types.is_numeric_dtype(df[col]):
                        suggestions.append({
                            'action': f'Imputer "{col}" (numérique)',
                            'colonnes': col,
                            'impact': f"{missing_pct:.1f}% manquant ({missing_count} valeurs)",
                            'code': f"df['{col}'].fillna(df['{col}'].median(), inplace=True)"
                        })
                    else:
                        suggestions.append({
                            'action': f'Imputer "{col}" (catégoriel)',
                            'colonnes': col,
                            'impact': f"{missing_pct:.1f}% manquant ({missing_count} valeurs)",
                            'code': f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)  # ou fillna('INCONNU')"
                        })
            
            # Suggestion 5: Outliers excessifs
            outliers_high = {k: v for k, v in profil['outliers'].items() if v > 10}
            if outliers_high:
                for col, count in list(outliers_high.items())[:3]:
                    suggestions.append({
                        'action': f'Traiter outliers dans "{col}"',
                        'colonnes': col,
                        'impact': f"{count} outliers détectés",
                        'code': f"""# Option 1: Winsorisation (cap aux percentiles)
q1, q99 = df['{col}'].quantile([0.01, 0.99])
df['{col}'] = df['{col}'].clip(lower=q1, upper=q99)

# Option 2: Suppression
# df = df[~((df['{col}'] < q1 - 1.5*iqr) | (df['{col}'] > q3 + 1.5*iqr))]"""
                    })
            
            if suggestions:
                sugg_df = pd.DataFrame(suggestions)
                st.dataframe(sugg_df[['action', 'colonnes', 'impact']], use_container_width=True)
                
                with st.expander("📝 Voir le code de nettoyage"):
                    code = "# Code de nettoyage suggéré\nimport pandas as pd\n\n"
                    for s in suggestions:
                        code += f"# {s['action']}\n{s['code']}\n\n"
                    st.code(code, language='python')
                
                if st.button("🔧 Appliquer toutes les corrections", type="primary"):
                    df_cleaned = df.copy()
                    
                    # Appliquer les corrections
                    if profil['empty_columns']:
                        df_cleaned = df_cleaned.drop(columns=profil['empty_columns'])
                    if profil['constant_columns']:
                        df_cleaned = df_cleaned.drop(columns=profil['constant_columns'])
                    if profil['duplicate_rows'] > 0:
                        df_cleaned = df_cleaned.drop_duplicates()
                    
                    st.success(f"✅ Nettoyage appliqué! {len(df)} → {len(df_cleaned)} lignes, {df.shape[1]} → {df_cleaned.shape[1]} colonnes")
                    
                    # Download cleaned file
                    csv_cleaned = df_cleaned.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger le fichier nettoyé",
                        data=csv_cleaned,
                        file_name="data_cleaned.csv",
                        mime="text/csv"
                    )
            else:
                st.success("✅ Aucune correction automatique suggérée - vos données sont déjà de bonne qualité!")

            # ----------------------------------------------------
            # PDF Export
            # ----------------------------------------------------
            st.markdown("---")
            st.subheader("📄 Export du rapport")
            
            fig_bytes = [fig_to_bytes(fig1)]

            pdf_buffer = build_pdf(synthesis, profil, df, fig_bytes)

            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="data_quality_report.pdf">📄 Télécharger le rapport PDF</a>'
            st.markdown(href, unsafe_allow_html=True)


# ----------------------------------------------------
# Contact Page
# ----------------------------------------------------
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**GitHub :** https://github.com/Ssoumano")
