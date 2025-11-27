import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# -------------------------
# Configuration OpenAI
# -------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Fonctions utilitaires Data Quality
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

    # Score global
    miss_score = max(0, 100 - profil["missing_pct"].mean())
    dup_score = max(0, 100 - (profil["duplicate_rows"] / max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(outliers.values())) if outliers else 0))

    profil["global_score"] = round((miss_score*0.5 + dup_score*0.3 + out_score*0.2), 1)

    return profil


# -------------------------
# Synthèse automatique PRO
# -------------------------
def generate_synthese(profil):
    missing = profil["missing_pct"].mean()
    duplicates = profil["duplicate_rows"]
    outliers_total = sum(profil["outliers"].values())

    synthese = f"""
### 🧾 Synthèse générale de la qualité des données

- Le dataset contient **{profil['rows']} lignes** et **{profil['cols']} colonnes**.
- Le taux moyen de valeurs manquantes est de **{missing:.2f}%**.
- Nombre total de doublons détectés : **{duplicates}**.
- Nombre total d'outliers dans les colonnes numériques : **{outliers_total}**.
- Score global calculé : **{profil['global_score']}%**.

"""

    # Priorités
    prio = "### 🎯 Priorités recommandées\n"

    if missing > 20:
        prio += "- 🔴 **Haute priorité : réduire les valeurs manquantes (>20%)**\n"
    elif missing > 5:
        prio += "- 🟠 **Priorité moyenne : valeurs manquantes modérées (>5%)**\n"
    else:
        prio += "- 🟢 **Faible priorité : peu de valeurs manquantes**\n"

    if duplicates > 0:
        prio += "- 🔴 **Supprimer les doublons détectés**\n"
    else:
        prio += "- 🟢 Aucun doublon détecté\n"

    if outliers_total > 20:
        prio += "- 🟠 **Analyser les colonnes contenant beaucoup d'outliers**\n"
    else:
        prio += "- 🟢 Outliers limités\n"

    return synthese + "\n" + prio


# -------------------------
# OpenAI : Tests complémentaires + Explications PRO
# -------------------------
def openai_suggest_tests(df):
    schema = ""
    for col in df.columns:
        schema += f"- {col}: {str(df[col].head().tolist())[:80]}...\n"

    prompt = f"""
    Analyse le schéma et génère :

    1. Une liste de tests de data quality avancés adaptés au dataset
    2. Une explication simple de chaque test
    3. Les risques associés si le test échoue
    4. Les recommandations de correction

    SCHÉMA :
    {schema}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert data quality senior."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -------------------------
# Interface Streamlit
# -------------------------
st.set_page_config(page_title="Data Quality App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Testez la qualité de vos données",
    "Contact"
])

info_style = """
    <span style='color:#888; font-size:14px; cursor:pointer; margin-left:4px;' title='{txt}'>ℹ️</span>
"""

# ============================
# PAGE : DATA QUALITY
# ============================
if page == "Testez la qualité de vos données":
    st.title("📊 Dashboard professionnel de Qualité des Données")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        df = load_dataframe(uploaded_file)

        if df is not None:
            profil = profile_data_quality(df)

            # ============================
            # KPI CARDS
            # ============================
            st.markdown("## ⭐ Indicateurs clés")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Score global", f"{profil['global_score']}%")
            col2.metric("Valeurs manquantes", int(profil["missing_count"].sum()))
            col3.metric("Doublons", profil["duplicate_rows"])
            col4.metric("Colonnes vides/constantes", len(profil["empty_columns"]) + len(profil["constant_columns"]))

            # ============================
            # Synthèse pro
            # ============================
            st.markdown(generate_synthese(profil))

            # ============================
            # Aperçu DataFrame
            # ============================
            st.subheader("👀 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # ============================
            # Heatmap Outliers (PRO)
            # ============================
            st.subheader("⚠️ Heatmap – Outliers détectés")

            outliers_df = (
                pd.DataFrame(profil["outliers"], index=["outliers"]).T
                .sort_values("outliers", ascending=False)
            )

            fig, ax = plt.subplots(figsize=(8, max(2, len(outliers_df) * 0.4)))

            sns.heatmap(
                outliers_df,
                annot=True,
                fmt="d",
                cmap="Reds",
                linewidths=.5,
                linecolor="white",
                cbar_kws={"label": "Niveau d’anomalies"},
                ax=ax
            )
            st.pyplot(fig)

            # ============================
            # Stats numériques
            # ============================
            st.subheader("📈 Statistiques numériques")
            st.dataframe(profil["numeric_stats"])

            # ============================
            # OPENAI : Tests complémentaires pro
            # ============================
            st.subheader("🤖 Tests de data quality avancés (OpenAI)")
            st.write(openai_suggest_tests(df))


# ============================
# CONTACT
# ============================
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**Github :** https://github.com/Ssoumano")
