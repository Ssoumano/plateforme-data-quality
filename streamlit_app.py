# app.py (VERSION PREMIUM)
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

# Try import OpenAI client (new 1.0+ style)
openai_available = True
try:
    from openai import OpenAI
except Exception:
    openai_available = False

# Try pdfkit for HTML->PDF conversion (optional)
try:
    import pdfkit
    pdfkit_available = True
except Exception:
    pdfkit_available = False

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Data Quality Platform — PRO", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dq_app")

# -----------------------------------------------------------------------------
# Styling (PowerBI-like tiles, small animations, info icons)
# -----------------------------------------------------------------------------
POWERBI_CSS = """
<style>
/* Layout and container */
section.main > div.block-container { max-width: 1400px; }

/* KPI tile base */
.kpi-tile {
  border-radius: 12px;
  padding: 16px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  box-shadow: 0 6px 22px rgba(0,0,0,0.08);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi-tile:hover { transform: translateY(-6px); box-shadow: 0 12px 30px rgba(0,0,0,0.12); }
.kpi-left { display:block; }
.kpi-label { color: rgba(0,0,0,0.6); font-size:13px; }
.kpi-value { font-weight:700; font-size:26px; margin-top:6px; }
.kpi-sub { font-size:12px; color: rgba(0,0,0,0.55); margin-top:6px; }

/* small info icon */
.kpi-info { font-size:14px; margin-left:8px; color:#666; }

/* charts */
.chart-card { padding: 12px; background: #fff; border-radius:10px; box-shadow: 0 6px 18px rgba(0,0,0,0.04); }

/* tooltip via title attr works well */
</style>
"""
st.markdown(POWERBI_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilities & Robust helpers
# -----------------------------------------------------------------------------
def safe_sum(obj: Any) -> int:
    """Return integer sum across common container types."""
    try:
        if isinstance(obj, dict):
            return int(sum(obj.values()))
        if isinstance(obj, pd.Series):
            return int(obj.sum())
        if isinstance(obj, (list, tuple, np.ndarray)):
            return int(sum(obj))
        # fallback: try cast to int
        return int(obj)
    except Exception:
        return 0

def safe_get(d: Dict, key, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default

def safe_to_datetime(series: pd.Series, threshold=0.8) -> bool:
    """Return True if series looks like datetime (>= threshold non-null after conversion)."""
    try:
        conv = pd.to_datetime(series, errors="coerce")
        return (conv.notna().sum() / max(1, len(series))) >= threshold
    except Exception:
        return False

def compute_col_types(df: pd.DataFrame) -> Dict[str, str]:
    """Simple automatic type detection per column."""
    types = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        if s.isna().all():
            types[col] = "empty"
            continue
        if safe_to_datetime(s):
            types[col] = "datetime"
            continue
        if pd.api.types.is_numeric_dtype(s):
            # numeric but maybe categorical (low cardinality)
            if s.nunique(dropna=True) / max(1, n) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "numeric"
            continue
        # not numeric: check binary / categorical / text
        nun = s.nunique(dropna=True)
        if nun == 2:
            types[col] = "binary"
        elif nun / max(1, n) < 0.05:
            types[col] = "categorical"
        else:
            types[col] = "text"
    return types

# -----------------------------------------------------------------------------
# Data profiling (robust)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def profile_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict of profiling metrics. Robust to datasets with no numeric columns."""
    profil = {}
    profil["rows"] = int(df.shape[0])
    profil["cols"] = int(df.shape[1])

    # missing counts and % (Series)
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    profil["missing_count"] = missing_count.to_dict()
    profil["missing_pct"] = missing_pct.to_dict()

    # dtypes
    profil["dtypes"] = df.dtypes.astype(str).to_dict()

    # constant / empty columns
    profil["constant_columns"] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    profil["empty_columns"] = [c for c in df.columns if df[c].dropna().shape[0] == 0]

    # duplicate rows
    profil["duplicate_rows"] = int(df.duplicated().sum())

    # numeric stats (if exists)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        profil["numeric_stats"] = {}
        profil["outliers"] = {}
    else:
        # ensure describe() safe
        try:
            stats = numeric.describe().T
            profil["numeric_stats"] = stats.to_dict(orient="index")
        except Exception:
            profil["numeric_stats"] = {}

        # outliers via IQR
        outliers = {}
        for col in numeric.columns:
            x = numeric[col].dropna()
            if x.empty:
                outliers[col] = 0
                continue
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers[col] = int(((x < lower) | (x > upper)).sum())
        profil["outliers"] = outliers

    # global score - robust when no numeric/outliers
    missing_mean = np.mean(list(profil["missing_pct"].values())) if profil["missing_pct"] else 0
    miss_score = max(0, 100 - missing_mean)
    dup_score = max(0, 100 - (profil["duplicate_rows"] / max(1, profil["rows"])) * 100)
    out_values = list(profil["outliers"].values()) if profil.get("outliers") else []
    out_score = max(0, 100 - np.mean(out_values)) if len(out_values)>0 else 100

    profil["global_score"] = round((miss_score * 0.5 + dup_score * 0.3 + out_score * 0.2), 1)

    return profil

# -----------------------------------------------------------------------------
# OpenAI helpers (compatible with openai>=1.0 , tries new "responses" then legacy)
# -----------------------------------------------------------------------------
def get_openai_client_from_secrets():
    if not openai_available:
        return None
    key = None
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        # optionally user can set via env or pass None (OpenAI client might still work if configured)
        key = None
    try:
        client = OpenAI(api_key=key) if key is not None else OpenAI()
        return client
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}")
        return None

def openai_generate_text(client: Any, prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """Try responses API (preferred), fallback to chat.completions if present."""
    if client is None:
        return "OpenAI non configuré."
    # Try responses.create (new API)
    try:
        # using messages format inside 'input' for Responses API
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # New responses API returns output array; extract text
        output_texts = []
        for item in resp.output:
            # item may have 'content' list of dicts with 'text'
            if isinstance(item, dict) and "content" in item:
                for c in item["content"]:
                    if "text" in c:
                        output_texts.append(c["text"])
            elif isinstance(item, str):
                output_texts.append(item)
        if output_texts:
            return "\n".join(output_texts)
        # Fallback to resp.output_text if available
        if hasattr(resp, "output_text"):
            return resp.output_text
        return str(resp)
    except Exception as e:
        logger.info(f"responses.create failed or not available: {e}")
        # fallback: try chat.completions (older)
        try:
            if hasattr(client, "chat"):
                resp2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"Tu es un expert en data quality."},
                              {"role":"user","content":prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                # extract text
                text = resp2.choices[0].message.content
                return text
        except Exception as e2:
            logger.warning(f"Fallback chat.completions failed: {e2}")
            return f"Erreur OpenAI (voir logs): {e2}"
    return "Erreur OpenAI inconnue."

def openai_synthesis_for_profile(client: Any, df: pd.DataFrame, profil: Dict[str, Any]) -> str:
    # Compose compact prompt (limit columns)
    sample_cols = list(df.columns[:15])
    schema_desc = ""
    for c in sample_cols:
        pct = profil["missing_pct"].get(c, 0)
        dtype = profil["dtypes"].get(c, str(df[c].dtype))
        uniq = int(df[c].nunique(dropna=True))
        schema_desc += f"- {c} ({dtype}): {pct:.1f}% missing, {uniq} unique\n"

    missing_mean = np.mean(list(profil["missing_pct"].values())) if profil["missing_pct"] else 0
    prompt = f"""
Tu es consultant senior en Data Quality. Analyse ce dataset.

Lignes: {profil['rows']}, Colonnes: {profil['cols']}
Score global: {profil['global_score']}%
Miss% moy.: {missing_mean:.2f}%
Doublons: {profil['duplicate_rows']}
Colonnes constantes: {profil['constant_columns']}

Extrait du schéma:
{schema_desc}

Réponds strictement au format Markdown:
### Synthèse Professionnelle
(10-15 lignes, impact business + risques)

### Tableau de Priorisation
| Priorité | Problème | Colonnes concernées | Impact | Recommandation |
|---|---|---|---|---|
(donne 5-7 lignes)

### Quick Wins
1) ...
2) ...
"""
    return openai_generate_text(client, prompt, max_tokens=2000, temperature=0.15)

def openai_suggest_tests_for_schema(client: Any, df: pd.DataFrame, profil: Dict[str, Any], col_types: Dict[str,str]) -> str:
    sample_cols = list(df.columns[:15])
    schema_desc = ""
    for c in sample_cols:
        pct = profil["missing_pct"].get(c, 0)
        ctype = col_types.get(c, "unknown")
        uniq = int(df[c].nunique(dropna=True))
        schema_desc += f"- {c} ({ctype}): {pct:.1f}% missing, {uniq} unique\n"

    prompt = f"""
Tu es expert en tests de data quality. Propose 8 à 12 tests précis et actionnables pour le schéma suivant:

{schema_desc}

Pour chaque test, fournis:
Titre
Objectif
Colonnes à tester
Méthode (ex: SQL/Pandas)
Critère de succès (seuil)

Couvre: complétude, validité, cohérence, unicité, exactitude.
"""
    return openai_generate_text(client, prompt, max_tokens=1800, temperature=0.2)

# -----------------------------------------------------------------------------
# Chart helpers (premium visuals)
# -----------------------------------------------------------------------------
def fig_to_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def create_outliers_heatmap(profil: Dict[str, Any]):
    out = profil.get("outliers", {})
    if not out:
        return None
    df_out = pd.DataFrame.from_dict(out, orient="index", columns=["outliers"]).sort_values("outliers", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(df_out))))
    sns.heatmap(df_out, annot=True, fmt="d", cmap="Reds", linewidths=.6, linecolor="white", ax=ax, cbar_kws={'label':'# outliers'})
    ax.set_title("Outliers détectés par colonne (IQR)")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    return fig

def create_missing_bar(profil: Dict[str, Any], top_n=10):
    mp = pd.Series(profil.get("missing_pct", {}))
    if mp.empty or mp.sum() == 0:
        return None
    mp = mp.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['#F44336' if v>50 else '#FFC107' if v>20 else '#4CAF50' for v in mp.values]
    mp.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Pourcentage manquant (%)")
    ax.invert_yaxis()
    for i,v in enumerate(mp.values):
        ax.text(v + max(0.5, v*0.02), i, f"{v:.1f}%", va='center', fontsize=9)
    plt.tight_layout()
    return fig

def create_missing_corr_heatmap(df: pd.DataFrame):
    # correlation matrix of missingness (0/1)
    mm = df.isna().astype(int)
    if mm.sum().sum() == 0:
        return None
    corr = mm.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.4, cbar_kws={"shrink":0.7}, ax=ax)
    ax.set_title("Corrélation des patterns de valeurs manquantes")
    plt.tight_layout()
    return fig

def create_types_pie(col_types: Dict[str,str]):
    s = pd.Series(list(col_types.values())).value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    s.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, colors=['#2196F3','#4CAF50','#FFC107','#F44336','#9C27B0'])
    ax.set_ylabel("")
    ax.set_title("Distribution des types de colonnes")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# PDF / HTML export
# -----------------------------------------------------------------------------
def generate_html_report(synthesis_markdown: str, profil: Dict[str,Any], charts_bytes: Dict[str, bytes]) -> str:
    """Return HTML string for the report. charts_bytes: name->bytes (PNG) - base64 them."""
    def img_b64(b):
        return base64.b64encode(b.getvalue()).decode() if hasattr(b, "getvalue") else base64.b64encode(b).decode()

    rows_meta = f"""
    <ul>
      <li>Lignes: {profil['rows']:,}</li>
      <li>Colonnes: {profil['cols']}</li>
      <li>Score global: {profil['global_score']}%</li>
      <li>Valeurs manquantes totales: {safe_sum(profil['missing_count'])}</li>
      <li>Doublons: {profil['duplicate_rows']}</li>
    </ul>
    """
    imgs_html = ""
    for k, b in charts_bytes.items():
        imgs_html += f"<h4>{k}</h4><img src='data:image/png;base64,{img_b64(b)}' style='max-width:100%;height:auto;'/>"

    html = f"""
    <html><head><meta charset="utf-8"><title>Rapport Data Quality</title></head><body style="font-family:Arial,Helvetica,sans-serif">
    <h1>Rapport Data Quality</h1>
    <h3>Métadonnées</h3>
    {rows_meta}
    <h3>Synthèse</h3>
    <div>{synthesis_markdown.replace('\\n','<br/>')}</div>
    <hr/>
    {imgs_html}
    <footer><small>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></footer>
    </body></html>
    """
    return html

def prepare_download_link_bytes(bytes_buf: io.BytesIO, filename: str, label: str):
    b64 = base64.b64encode(bytes_buf.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href

def export_pdf_from_html(html: str) -> Optional[bytes]:
    """Try to produce PDF bytes using pdfkit if available."""
    if not pdfkit_available:
        return None
    try:
        pdf_bytes = pdfkit.from_string(html, False)
        return pdf_bytes
    except Exception as e:
        logger.warning(f"pdfkit conversion failed: {e}")
        return None

# -----------------------------------------------------------------------------
# UI page layout
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Tableau de bord", "Contact"])

if page == "Contact":
    st.title("Contact")
    st.markdown("""
**Nom :** SOUMANO Seydou  
**E-mail :** soumanoseydou@icloud.com  
**Téléphone :** +33 6 64 67 88 87  
**LinkedIn :** https://linkedin.com/in/seydou-soumano  
**GitHub :** https://github.com/Ssoumano
""")
else:
    # Main dashboard
    st.title("📊 Data Quality Platform — PRO")
    st.write("Importe un fichier (CSV/XLSX). L'app analyse automatiquement la qualité, propose synthèse et tests IA, et génère un rapport.")

    # OpenAI client
    openai_client = get_openai_client_from_secrets() if openai_available else None
    if openai_client is None:
        st.info("OpenAI non configuré — ajoute OPENAI_API_KEY dans les secrets pour activer les fonctionnalités IA.")

    uploaded_file = st.file_uploader("Importer un fichier (CSV / XLSX)", type=["csv","xlsx","xls"])
    if uploaded_file is None:
        st.info("Téléverse un fichier pour démarrer l'analyse.")
    else:
        # Try load dataframe robustly
        try:
            # Try pandas to detect separator and read
            name_l = uploaded_file.name.lower()
            raw = uploaded_file.getvalue()
            if name_l.endswith(".csv"):
                sample = raw[:4096].decode(errors="ignore")
                sep = "," if "," in sample else ";" if ";" in sample else ","
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            elif name_l.endswith((".xls", ".xlsx")):
                df = pd.read_excel(io.BytesIO(raw))
            else:
                df = pd.read_csv(io.BytesIO(raw), engine="python")
        except Exception as e:
            logger.exception("Failed to load file")
            st.error(f"Impossible de lire le fichier: {e}")
            df = None

        if df is None:
            st.stop()

        # Limit df for display but keep full df for analysis
        display_limit = 300
        st.subheader("Aperçu du dataset")
        try:
            st.dataframe(df.head(display_limit))
        except Exception:
            st.write("Aperçu non disponible (dataset volumineux).")

        # Profiling (cached)
        with st.spinner("Profiling des données..."):
            try:
                profil = profile_data_quality(df)
            except Exception as e:
                logger.exception("Profiling failed")
                st.error(f"Erreur lors du profiling: {e}")
                profil = {
                    "rows": len(df),
                    "cols": df.shape[1],
                    "missing_count": df.isna().sum().to_dict(),
                    "missing_pct": (df.isna().mean()*100).round(2).to_dict(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "constant_columns": [],
                    "empty_columns": [],
                    "duplicate_rows": int(df.duplicated().sum()),
                    "outliers": {},
                    "numeric_stats": {},
                    "global_score": 0
                }

        # Column type detection
        col_types = compute_col_types(df)

        # KPI Tiles
        c1, c2, c3, c4 = st.columns(4, gap="large")
        # tile 1
        info_score = "Indice synthétique : missing (50%), duplicates (30%), outliers (20%)."
        with c1:
            st.markdown(f"""
            <div class="kpi-tile" title="{info_score}" style="background:#F2C811;">
                <div class="kpi-left">
                    <div class="kpi-label">Score global <span class="kpi-info">ℹ️</span></div>
                    <div class="kpi-value">{profil['global_score']}%</div>
                    <div class="kpi-sub">Indice synthétique</div>
                </div>
                <div style="font-size:36px;">📊</div>
            </div>
            """, unsafe_allow_html=True)

        # tile 2
        with c2:
            missing_total = safe_sum(profil.get("missing_count", {}))
            st.markdown(f"""
            <div class="kpi-tile" title="Nombre total de cellules vides / NA" style="background:#ffffff;">
                <div class="kpi-left">
                    <div class="kpi-label">Valeurs manquantes <span class="kpi-info">ℹ️</span></div>
                    <div class="kpi-value">{missing_total}</div>
                    <div class="kpi-sub">Total de NA</div>
                </div>
                <div style="font-size:36px;">❗</div>
            </div>
            """, unsafe_allow_html=True)

        # tile 3
        with c3:
            st.markdown(f"""
            <div class="kpi-tile" title="Nombre de lignes strictement dupliquées" style="background:#118DFF;color:white;">
                <div class="kpi-left">
                    <div class="kpi-label">Doublons <span class="kpi-info">ℹ️</span></div>
                    <div class="kpi-value">{profil.get('duplicate_rows', 0)}</div>
                    <div class="kpi-sub">Lignes dupliquées</div>
                </div>
                <div style="font-size:36px;">📑</div>
            </div>
            """, unsafe_allow_html=True)

        # tile 4
        with c4:
            cols_vide_const = len(profil.get("empty_columns", [])) + len(profil.get("constant_columns", []))
            st.markdown(f"""
            <div class="kpi-tile" title="Colonnes sans variance (constantes) ou totalement vides" style="background:#f6f6f6;">
                <div class="kpi-left">
                    <div class="kpi-label">Colonnes vides/constantes <span class="kpi-info">ℹ️</span></div>
                    <div class="kpi-value">{cols_vide_const}</div>
                    <div class="kpi-sub">Colonnes sans variance</div>
                </div>
                <div style="font-size:36px;">📦</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Two heatmaps: outliers heatmap (if numeric) + missing corr heatmap
        st.subheader("Visualisations avancées")

        colA, colB = st.columns([1,1], gap="large")
        out_fig = create_outliers_heatmap(profil)
        missbar_fig = create_missing_bar(profil)
        misscorr_fig = create_missing_corr_heatmap(df)
        types_fig = create_types_pie(col_types)

        with colA:
            if out_fig:
                st.markdown("**Heatmap — Outliers (IQR)**  ℹ️")
                st.pyplot(out_fig)
            else:
                st.info("Aucune colonne numérique / outliers détectés pour cette dataset.")

        with colB:
            if missbar_fig:
                st.markdown("**Top colonnes — Valeurs manquantes**  ℹ️")
                st.pyplot(missbar_fig)
            else:
                st.info("Aucune valeur manquante significative détectée.")

        st.markdown("---")
        # Missing correlation heatmap (full)
        if misscorr_fig:
            st.subheader("Corrélation des patterns de valeurs manquantes")
            st.pyplot(misscorr_fig)
            st.info("Valeurs proches de 1 signifient que les colonnes manquent souvent ensemble.")
        else:
            st.info("Aucune corrélation de missingness (pas de NA détectées).")

        # types pie
        st.markdown("---")
        st.subheader("Typologie des colonnes détectées")
        st.pyplot(types_fig)

        # Column profiling interactive (premium)
        st.markdown("---")
        st.subheader("Profiling détaillé par colonne")
        col_selected = st.selectbox("Sélectionne une colonne", df.columns.tolist())
        st.write("Type détecté:", col_types.get(col_selected, "unknown"))
        st.write("Valeurs manquantes:", f"{df[col_selected].isna().sum()} ({df[col_selected].isna().mean()*100:.1f}%)")
        st.write("Valeurs uniques:", df[col_selected].nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(df[col_selected]):
            figc, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
            df[col_selected].dropna().hist(bins=30, ax=ax1, color="#118DFF", edgecolor="white")
            ax1.set_title("Distribution")
            df[col_selected].dropna().plot.box(ax=ax2)
            ax2.set_title("Boxplot")
            st.pyplot(figc)
            st.write(df[col_selected].describe().to_frame())
        else:
            st.write(df[col_selected].value_counts().head(30).to_frame())

        # Suggestions cleaning (quick wins)
        st.markdown("---")
        st.subheader("Suggestions de nettoyage automatiques (Quick Wins)")
        suggestions = []
        if profil.get("empty_columns"):
            suggestions.append({"action":"Supprimer colonnes vides", "columns": profil["empty_columns"], "code":f"df = df.drop(columns={profil['empty_columns']})"})
        if profil.get("constant_columns"):
            suggestions.append({"action":"Supprimer constantes", "columns": profil["constant_columns"], "code":f"df = df.drop(columns={profil['constant_columns']})"})
        if profil.get("duplicate_rows",0) > 0:
            suggestions.append({"action":"Supprimer doublons", "columns":"Toutes", "code":"df = df.drop_duplicates()"})
        # missing imputations per column (top ones)
        missing_series = pd.Series(profil.get("missing_pct", {}))
        high_missing = missing_series[missing_series > 0].sort_values(ascending=False).head(10)
        for col_name, pct in high_missing.items():
            if pd.api.types.is_numeric_dtype(df[col_name]):
                code = f"df['{col_name}'].fillna(df['{col_name}'].median(), inplace=True)"
            else:
                code = f"df['{col_name}'].fillna(df['{col_name}'].mode()[0] if not df['{col_name}'].mode().empty else 'INCONNU', inplace=True)"
            suggestions.append({"action":f"Imputer {col_name}", "columns":[col_name], "code":code})

        if suggestions:
            for s in suggestions:
                st.write(f"- **{s['action']}** — Colonnes: {s['columns'] if 'columns' in s else 'N/A'}")
            with st.expander("Voir le code des quick wins"):
                code_text = "# Suggestions de nettoyage\n"
                for s in suggestions:
                    code_text += f"# {s['action']}\n{ s['code'] }\n\n"
                st.code(code_text, language="python")
        else:
            st.success("Aucune action de nettoyage détectée — dataset propre !")

        # -----------------------------------------------------------------------------
        # OpenAI synthèse & tests
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Synthèse & Priorités (IA)")
        col_ia1, col_ia2 = st.columns([1,3])
        with col_ia1:
            if openai_client is None:
                st.info("Active OpenAI (OPENAI_API_KEY dans les secrets) pour avoir la synthèse IA.")
            else:
                if st.button("Générer la synthèse IA"):
                    with st.spinner("Appel OpenAI en cours..."):
                        try:
                            synth_text = openai_synthesis_for_profile(openai_client, df, profil)
                            st.markdown(synth_text)
                        except Exception as e:
                            st.error(f"Erreur OpenAI: {e}")
        with col_ia2:
            st.info("La synthèse IA produit une synthèse professionnelle, un tableau de priorisation et des quick wins.")

        # Tests suggestions
        st.markdown("---")
        st.subheader("Tests complémentaires suggérés (IA)")
        with st.spinner("Génération des tests recommandés..."):
            if openai_client is None:
                st.info("OpenAI non configuré → impossible de générer les tests IA.")
            else:
                if st.button("Générer les tests IA"):
                    try:
                        tests_text = openai_suggest_tests_for_schema(openai_client, df, profil, col_types)
                        st.markdown(tests_text)
                    except Exception as e:
                        st.error(f"Erreur OpenAI tests: {e}")

        # -----------------------------------------------------------------------------
        # Export / PDF generation (HTML -> PDF via pdfkit if available)
        # -----------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Export / Rapport")

        # Prepare charts bytes
        charts = {}
        if out_fig:
            charts["Outliers heatmap"] = fig_to_bytes(out_fig)
        if missbar_fig:
            charts["Missing top cols"] = fig_to_bytes(missbar_fig)
        if misscorr_fig:
            charts["Missing correlation"] = fig_to_bytes(misscorr_fig)
        if types_fig:
            charts["Types distribution"] = fig_to_bytes(types_fig)

        # Get synthesis text if already generated else short local synthesis
        synth_local = locals().get("synth_text", None)
        if 'synth_text' in locals() and synth_text:
            final_synth = synth_text
        else:
            # create small local synthesis fallback
            final_synth = f"Dataset: {profil['rows']:,} lignes × {profil['cols']} colonnes. Score global: {profil['global_score']}%. Top missing: {', '.join(list(pd.Series(profil['missing_pct']).sort_values(ascending=False).head(3).index)) if profil['missing_pct'] else 'N/A'}"

        html_report = generate_html_report(final_synth, profil, charts)

        # Offer download HTML always
        b64_html = base64.b64encode(html_report.encode()).decode()
        st.markdown(f'<a href="data:text/html;base64,{b64_html}" download="report_data_quality.html">📄 Télécharger le rapport (HTML)</a>', unsafe_allow_html=True)

        # Offer PDF if pdfkit available
        if pdfkit_available:
            if st.button("📄 Générer PDF (wkhtmltopdf)"):
                with st.spinner("Conversion HTML → PDF en cours..."):
                    try:
                        pdf_bytes = export_pdf_from_html(html_report)
                        if pdf_bytes:
                            st.success("PDF prêt.")
                            st.download_button(label="📥 Télécharger le rapport PDF", data=pdf_bytes, file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                        else:
                            st.error("La conversion PDF a échoué (voir logs).")
                    except Exception as e:
                        st.error(f"Erreur conversion PDF: {e}")
        else:
            st.info("pdfkit/wkhtmltopdf non présent sur l'environnement — utilise le téléchargement HTML ou installe wkhtmltopdf pour PDF direct.")

        # End of dataset flow

# -----------------------------------------------------------------------------
# End of app
# -----------------------------------------------------------------------------
