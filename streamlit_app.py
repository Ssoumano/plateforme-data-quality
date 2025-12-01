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

# -------------------------
# CONFIG OPENAI (via secrets recommended)
# -------------------------
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    OPENAI_API_KEY = None
    client = None

# -------------------------
# Utility: separator detection & load
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

# -------------------------
# Profiling functions
# -------------------------
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

    # outliers IQR
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

    # global score
    miss_score = max(0, 100 - profil["missing_pct"].mean())
    dup_score = max(0, 100 - (profil["duplicate_rows"]/max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(outliers.values())) if outliers else 0))
    profil["global_score"] = round((miss_score*0.5 + dup_score*0.3 + out_score*0.2), 1)

    return profil

# -------------------------
# OpenAI helpers (if configured)
# -------------------------
def openai_generate_synthesis(df, profil):
    if client is None:
        return "OpenAI non configuré. Ajoute OPENAI_API_KEY dans les secrets Streamlit."
    schema = ""
    for col in df.columns:
        schema += f"- {col}: {str(df[col].head(5).tolist())[:120]}...\n"

    prompt = f"""
    Tu es consultant expert en Data Quality.
    Données : Lignes {profil['rows']}, Colonnes {profil['cols']}.
    Détail : missing_pct_mean={profil['missing_pct'].mean()}, duplicates={profil['duplicate_rows']}, outliers={profil['outliers']}, constant_columns={profil['constant_columns']}

    Donne :
    1) Une synthèse professionnelle (10-15 lignes).
    2) Un tableau Priorité | Problème | Colonnes | Recommandation (Priorité: Haute/Moyenne/Basse).
    3) 5 quick wins.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Tu es un consultant senior expert en qualité des données."},
            {"role":"user","content": prompt}
        ],
        max_tokens=900,
        temperature=0.2
    )
    return resp.choices[0].message.content

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
            {"role":"system","content":"Expert en data quality."},
            {"role":"user","content": prompt}
        ],
        max_tokens=600,
        temperature=0.2
    )
    return resp.choices[0].message.content

# -------------------------
# PDF generation (ReportLab)
# -------------------------
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

    # Metadata
    meta = [
        ["Lignes", profil['rows']],
        ["Colonnes", profil['cols']],
        ["Score global", f"{profil['global_score']}%"],
        ["Valeurs manquantes (total)", int(profil['missing_count'].sum())],
        ["Doublons", profil['duplicate_rows']],
    ]
    t = Table(meta, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('TEXTCOLOR',(0,0),(-1,-1), colors.black),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(t)
    story.append(Spacer(1,12))

    # Report text (synthese)
    story.append(Paragraph("Synthèse", styles["Heading2"]))
    for para in report_text.split("\n\n"):
        story.append(Paragraph(para.replace("\n","<br/>"), styles["BodyText"]))
        story.append(Spacer(1,6))

    story.append(PageBreak())

    # Add figures (each image on page)
    for idx, figb in enumerate(figs_bytes_list):
        story.append(Paragraph(f"Figure {idx+1}", styles["Heading2"]))
        img = Image.open(figb)
        # resize for A4 width
        max_w = A4[0] - 4*cm
        w, h = img.size
        ratio = max_w / w
        new_w = max_w
        new_h = h * ratio
        img_buf = io.BytesIO()
        img.save(img_buf, format='PNG')
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=new_w, height=new_h)
        story.append(rl_img)
        story.append(Spacer(1,12))

    # Detailed tables: top problematic columns (missing & outliers)
    story.append(PageBreak())
    story.append(Paragraph("Détails - Colonnes", styles["Heading2"]))
    # Missing table (top 20)
    mdf = profil['missing_pct'].sort_values(ascending=False).head(20)
    missing_table = [["Colonne", "Missing %", "Missing Count"]]
    for col in mdf.index:
        missing_table.append([col, f"{profil['missing_pct'][col]}%", int(profil['missing_count'][col])])
    t2 = Table(missing_table, repeatRows=1, hAlign='LEFT')
    t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0), colors.HexColor('#F2C811')),
                            ('GRID',(0,0),(-1,-1),0.3, colors.grey)]))
    story.append(t2)

    # Numeric stats sample
    story.append(PageBreak())
    story.append(Paragraph("Statistiques numériques (extrait)", styles["Heading2"]))
    ns = profil['numeric_stats'].reset_index().head(50)
    data_table = [ns.columns.tolist()] + ns.fillna("").values.tolist()
    t3 = Table(data_table, repeatRows=1)
    t3.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.3, colors.grey),
                            ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#118DFF'))]))
    story.append(t3)

    doc.build(story)
    buf.seek(0)
    return buf

# -------------------------
# Styling Power BI (partial)
# -------------------------
POWERBI_CSS = """
<style>
/* background */
section.main > div.block-container { max-width: 1400px; }

/* Title style */
h1 { font-family: "Segoe UI", sans-serif; }

/* KPI tile fonts */
.kpi-val { font-size: 28px; font-weight:700; color:#111; }
.kpi-label { font-size: 12px; color:#222; }

/* small spacing tweak */
div.block-container { padding-top: 18px; padding-left:18px; padding-right:18px; }
</style>
"""

st.set_page_config(page_title="Data Quality App", layout="wide")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Testez la qualité de vos données", "Contact"])

# -------------------------
# Page: Data Quality
# -------------------------
if page == "Testez la qualité de vos données":
    st.markdown(POWERBI_CSS, unsafe_allow_html=True)
    st.title("📊 Data Quality Dashboard (Power BI style tiles)")

    if OPENAI_API_KEY is None:
        st.warning("OpenAI non configuré. Va dans Settings → Secrets et ajoute OPENAI_API_KEY pour activer les synthèses.")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        df = load_dataframe(uploaded_file)
        if df is None:
            st.error("Impossible de lire le fichier.")
        else:
            profil = profile_data_quality(df)

            # KPI tiles - PowerBI look (Option 2: only tiles & charts themed)
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

            # colors: Power BI palette
            yellow = "#F2C811"
            blue = "#118DFF"
            dark = "#2B2B2B"
            white = "#FFFFFF"

            c1, c2, c3, c4 = st.columns([1,1,1,1], gap="large")
            c1.markdown(tile_style.format(bg=yellow, text=dark, muted="#563C00",
                                          label="Score global", value=f"{profil['global_score']}%",
                                          subtitle="Indice synthétique", icon="📊"),
                        unsafe_allow_html=True)
            c2.markdown(tile_style.format(bg="#ffffff", text=dark, muted="#666",
                                          label="Valeurs manquantes", value=int(profil['missing_count'].sum()),
                                          subtitle="Total de cellules vides", icon="❗"),
                        unsafe_allow_html=True)
            c3.markdown(tile_style.format(bg=blue, text=white, muted="#DDF4FF",
                                          label="Doublons", value=profil['duplicate_rows'],
                                          subtitle="Lignes dupliquées", icon="📑"),
                        unsafe_allow_html=True)
            c4.markdown(tile_style.format(bg="#f6f6f6", text=dark, muted="#666",
                                          label="Colonnes vides/constantes",
                                          value=len(profil['empty_columns']) + len(profil['constant_columns']),
                                          subtitle="Colonnes sans variance", icon="📦"),
                        unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("👀 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # Heatmap outliers (styled)
            st.subheader("🔥 Heatmap – Outliers (IQR)")
            outlier_df = pd.DataFrame(profil['outliers'].values(), index=profil['outliers'].keys(), columns=["outliers"])
            outlier_df = outlier_df.sort_values("outliers", ascending=False)
            fig1, ax1 = plt.subplots(figsize=(8, max(2, len(outlier_df)*0.35)))
            sns.heatmap(outlier_df, annot=True, fmt="d", cmap=sns.color_palette("Reds", as_cmap=True),
                        linewidths=0.6, linecolor="white", cbar_kws={'label': 'Nombre d\'outliers'}, ax=ax1)
            ax1.set_title("Outliers détectés par colonne (IQR)", fontsize=13, pad=12)
            ax1.set_ylabel("")
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)
            st.pyplot(fig1)

            st.markdown("---")
            st.subheader("🧠 Synthèse détaillée & Priorités (OpenAI)")
            with st.spinner("Génération OpenAI..."):
                synthesis = openai_generate_synthesis(df, profil)
            st.markdown(synthesis)

            st.markdown("## 🧪 Tests complémentaires (OpenAI)")
            tests = openai_suggest_tests(df)
            st.write(tests)

            # Prepare figures for PDF: KPI as small image + heatmap
            # Create KPI image (simple) by rendering the tiles section to a PNG is complex;
            # We'll include the heatmap + a small table and the synthese as text in PDF.
            fig_bytes = []
            # outlier heatmap
            fh = fig_to_bytes(fig1)
            fig_bytes.append(fh)

            # build PDF (detailed)
            report_text = synthesis
            pdf_buffer = build_pdf(report_text, profil, df, fig_bytes)

            # Download button
            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="data_quality_report.pdf">📄 Télécharger le rapport complet (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)

# -------------------------
# Page Contact
# -------------------------
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**GitHub :** https://github.com/Ssoumano")
