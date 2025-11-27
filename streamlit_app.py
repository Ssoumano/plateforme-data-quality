import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# -------------------------
# CONFIG OPENAI
# -------------------------
OPENAI_API_KEY = "YOUR_OPENAI_KEY_HERE"
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Fonctions Data Quality
# -------------------------
def detect_separator(uploaded_file_bytes: bytes) -> str:
    sample = uploaded_file_bytes[:4096].decode(errors='ignore')
    for sep in [';', ',', '\t', '|']:
        if sep in sample:
            return sep
    return ','

def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith('.csv'):
        sep = detect_separator(data)
        return pd.read_csv(io.BytesIO(data), sep=sep, encoding='utf-8', engine='python')

    elif name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(data))

    else:
        return pd.read_csv(io.BytesIO(data), encoding='utf-8')


def profile_data_quality(df: pd.DataFrame) -> dict:
    profil = {}
    profil['rows'] = int(df.shape[0])
    profil['cols'] = int(df.shape[1])

    profil['missing_count'] = df.isna().sum()
    profil['missing_pct'] = (df.isna().mean() * 100).round(2)

    profil['dtypes'] = df.dtypes.astype(str)
    profil['constant_columns'] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    profil['empty_columns'] = [c for c in df.columns if df[c].dropna().shape[0] == 0]
    profil['duplicate_rows'] = int(df.duplicated().sum())

    numeric = df.select_dtypes(include=[np.number])
    profil['numeric_stats'] = numeric.describe().T

    # Outliers avec IQR
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

    # Score global
    miss_score = max(0, 100 - profil["missing_pct"].mean())
    dup_score = max(0, 100 - (profil["duplicate_rows"] / max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(outliers.values())) if outliers else 0))

    profil["global_score"] = round((miss_score*0.5 + dup_score*0.3 + out_score*0.2), 1)

    return profil


# -------------------------
# OpenAI – Synthèse & Priorités
# -------------------------
def openai_generate_synthesis(df, profil):
    """Synthèse détaillée (10-15 lignes) + priorités"""

    schema_desc = ""
    for col in df.columns:
        schema_desc += f"- {col} : {str(df[col].head().tolist())[:60]}...\n"

    prompt = f"""
    Tu es consultant expert en Data Quality.

    Voici le profil du dataset :

    - Lignes: {profil['rows']}
    - Colonnes: {profil['cols']}
    - Valeurs manquantes (% moyen): {profil['missing_pct'].mean()}
    - Colonnes avec NA: {profil['missing_count'][profil['missing_count']>0].index.tolist()}
    - Doublons: {profil['duplicate_rows']}
    - Outliers: {profil['outliers']}
    - Colonnes constantes: {profil['constant_columns']}
    - Types: {dict(profil['dtypes'])}

    Voici un aperçu des valeurs :
    {schema_desc}

    1️⃣ Rédige une synthèse détaillée de la qualité des données (10 à 15 lignes).
    Style professionnel, structuré, clair, comme un consultant data.

    2️⃣ Identifie les priorités d'amélioration dans un tableau formaté comme ceci :

    | Priorité | Problème | Colonnes concernées | Recommandation |
    |---------|----------|---------------------|----------------|

    Les priorités doivent être : Haute / Moyenne / Basse.

    3️⃣ Donne aussi une liste d’actions rapides (Quick Wins) en 5 points.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un consultant senior expert en qualité des données."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Erreur OpenAI : {e}"


def openai_suggest_tests(df):
    schema_description = ""
    for col in df.columns:
        schema_description += f"- {col}: {str(df[col].head().tolist())[:80]}...\n"

    prompt = f"""
    Analyse le schéma ci-dessous et propose 5 à 10 tests de data quality supplémentaires.

    SCHÉMA DES DONNÉES :
    {schema_description}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en data quality."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Erreur OpenAI : {e}"

# -------------------------
# Interface Streamlit
# -------------------------
st.set_page_config(page_title="Data Quality App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Testez la qualité de vos données", "Contact"])

info_icon = "<span style='color:#888;font-size:14px;margin-left:4px;cursor:pointer;'>ℹ️</span>"

# ---------------------------------------------------------------------
# PAGE : DATA QUALITY
# ---------------------------------------------------------------------
if page == "Testez la qualité de vos données":

    st.title("📊 Data Quality Dashboard")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        df = load_dataframe(uploaded_file)
        profil = profile_data_quality(df)

        # -------------------------
        # KPI Cards
        # -------------------------
        st.markdown("## ⭐ Indicateurs clés")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Score global", f"{profil['global_score']}%")
        col2.metric("Valeurs manquantes", int(profil["missing_count"].sum()))
        col3.metric("Doublons", profil["duplicate_rows"])
        col4.metric("Colonnes vides/constantes", len(profil["empty_columns"]) + len(profil["constant_columns"]))

        # -------------------------
        # Aperçu DataFrame
        # -------------------------
        st.subheader("Aperçu du DataFrame")
        st.dataframe(df.head(300))

        # -------------------------
        # Heatmap OUTLIERS PRO
        # -------------------------
        st.subheader("🔥 Heatmap des Outliers")

        outlier_df = pd.DataFrame({
            "colonne": list(profil["outliers"].keys()),
            "outliers": list(profil["outliers"].values())
        }).set_index("colonne")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(outlier_df, annot=True, fmt=".0f", cmap="Reds", linewidths=.5, ax=ax)
        ax.set_title("Niveau d'Outliers par Colonne", fontsize=12)
        st.pyplot(fig)

        # -------------------------
        # Synthèse & Priorités OpenAI
        # -------------------------
        st.markdown("## 🧠 Synthèse détaillée & Priorités (OpenAI)")
        with st.spinner("Analyse OpenAI en cours..."):
            synthesis = openai_generate_synthesis(df, profil)
        st.markdown(synthesis)

        # -------------------------
        # Suggestions de Tests
        # -------------------------
        st.markdown("## 🧪 Suggestions de tests complémentaires (OpenAI)")
        test_suggestions = openai_suggest_tests(df)
        st.write(test_suggestions)


# ---------------------------------------------------------------------
# PAGE CONTACT
# ---------------------------------------------------------------------
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**Github :** https://github.com/Ssoumano")
