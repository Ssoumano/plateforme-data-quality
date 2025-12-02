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


# ----------------------------------------------------
# Data Profiling
# ----------------------------------------------------
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

    # Global score
    miss_score = max(0, 100 - profil["missing_pct"].mean())
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
    for col in df.columns:
        schema += f"- {col}: {str(df[col].head(5).tolist())[:120]}...\n"

    prompt = f"""
    Tu es consultant expert en Data Quality.
    Données : Lignes {profil['rows']}, Colonnes {profil['cols']}.
    Détail : missing_pct_mean={profil['missing_pct'].mean()}, duplicates={profil['duplicate_rows']}, 
    outliers={profil['outliers']}, constant_columns={profil['constant_columns']}

    Donne :
    1) Une synthèse professionnelle (10-15 lignes).
    2) Un tableau Priorité | Problème | Colonnes | Recommandation.
    3) 5 quick wins.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant senior expert en qualité des données."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
        temperature=0.2
    )

    return clean_ai_text(resp.choices[0].message.content)


def openai_suggest_tests(df):
    if client is None:
        return "OpenAI non configuré."

    schema = ""
    for col in df.columns:
        schema += f"- {col}: {str(df[col].head(5).tolist())[:120]}...\n"

    prompt = f"Propose 5-10 tests de data quality adaptés au schéma suivant:\n{schema}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert en data quality."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
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
        ["Valeurs manquantes (total)", int(profil['missing_count'].sum())],
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
        df = load_dataframe(uploaded_file)

        if df is None:
            st.error("❌ Impossible de lire le fichier.")
        else:
            profil = profile_data_quality(df)

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
                label="Valeurs manquantes", value=int(profil["missing_count"].sum()),
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
            # DataFrame Preview
            # ----------------------------------------------------
            st.subheader("📄 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # ----------------------------------------------------
            # Outlier Heatmap
            # ----------------------------------------------------
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

            with st.spinner("Analyse en cours..."):
                synthesis = openai_generate_synthesis(df, profil)

            st.markdown(synthesis)

            # ----------------------------------------------------
            # Tests OpenAI
            # ----------------------------------------------------
            st.subheader("🧪 Tests complémentaires suggérés")
            tests = openai_suggest_tests(df)
            st.write(tests)

            # ----------------------------------------------------
            # PDF Export
            # ----------------------------------------------------
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



