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

def openai_suggest_tests(df):
    schema_description = ""

    for col in df.columns:
        schema_description += f"- {col}: {str(df[col].head().tolist())[:80]}...\n"

    prompt = f"""
    Analyse le schéma ci-dessous et propose des tests de data quality
    supplémentaires adaptés aux colonnes.

    SCHÉMA DES DONNÉES :
    {schema_description}

    Fournis une liste claire d'au moins 5 tests pertinents.
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
page = st.sidebar.radio("Aller à", [
    "Testez la qualité de vos données",
    "Contact"
])

# Style icône (i)
info_style = """
    <span style='color:#888; font-size:14px; cursor:pointer; margin-left:4px;' title='{txt}'>ℹ️</span>
"""

# ============================
# PAGE : DATA QUALITY
# ============================
if page == "Testez la qualité de vos données":
    st.title("📊 Dashboard de Qualité des Données")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        df = load_dataframe(uploaded_file)

        if df is not None:
            profil = profile_data_quality(df)

            # ============================
            # KPI CARDS + (i)
            # ============================
            st.markdown("## ⭐ Indicateurs clés")

            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(
                f"<b>Score global</b>{info_style.format(txt='Score basé sur NA (50%), doublons (30%) et outliers (20%).')}",
                unsafe_allow_html=True
            )
            col1.metric("", f"{profil['global_score']}%")

            col2.markdown(
                f"<b>Valeurs manquantes</b>{info_style.format(txt='Nombre total de valeurs manquantes détectées.')}",
                unsafe_allow_html=True
            )
            col2.metric("", int(profil["missing_count"].sum()))

            col3.markdown(
                f"<b>Doublons</b>{info_style.format(txt='Nombre de lignes complètement dupliquées.')}",
                unsafe_allow_html=True
            )
            col3.metric("", profil["duplicate_rows"])

            col4.markdown(
                f"<b>Colonnes vides/constantes</b>{info_style.format(txt='Colonnes sans données ou avec une seule valeur.')}",
                unsafe_allow_html=True
            )
            col4.metric("", len(profil["empty_columns"]) + len(profil["constant_columns"]))

            # ============================
            # Aperçu DataFrame
            # ============================
            st.subheader("👀 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # ============================
            # HEATMAP OUTLIERS (VERSION PRO)
            # ============================
            st.subheader("⚠️ Heatmap – Nombre d’outliers (IQR)")

            outliers_df = (
                pd.DataFrame(profil["outliers"], index=["outliers"]).T
                .sort_values("outliers", ascending=False)
            )

            fig, ax = plt.subplots(figsize=(8, max(2, len(outliers_df) * 0.4)))

            sns.heatmap(
                outliers_df,
                annot=True,
                fmt="d",
                cmap=sns.color_palette("Reds", as_cmap=True),
                linewidths=.5,
                linecolor="white",
                cbar_kws={"label": "Niveau d’anomalies"},
                ax=ax
            )

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("Outliers détectés par colonne (Méthode IQR)", fontsize=14, pad=12)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

            st.pyplot(fig)

            # ============================
            # Stats numériques
            # ============================
            st.subheader("📈 Statistiques numériques")
            st.dataframe(profil["numeric_stats"])

            # ============================
            # Suggestions OpenAI
            # ============================
            st.subheader("🤖 Suggestions de tests complémentaires (OpenAI)")
            st.write(openai_suggest_tests(df))

# ============================
# PAGE CONTACT
# ============================
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**Github :** https://github.com/Ssoumano")

