# app.py - HYBRID (regex + OpenAI option) Data Quality Platform
import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional

# Try import OpenAI client (new 1.0+ style). If not available, we will keep IA option disabled.
openai_available = True
try:
    from openai import OpenAI
except Exception:
    openai_available = False

# Optional pdf conversion
try:
    import pdfkit
    pdfkit_available = True
except Exception:
    pdfkit_available = False

# ----------------------------
# App config & styling
# ----------------------------
st.set_page_config(page_title="Data Quality — Hybrid", layout="wide")
POWERBI_CSS = """
<style>
section.main > div.block-container { max-width: 1400px; }
.kpi-tile { border-radius:12px; padding:12px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 6px 18px rgba(0,0,0,0.06);}
.kpi-left {display:block;}
.kpi-label {font-size:13px; color:rgba(0,0,0,0.6);}
.kpi-value {font-weight:700; font-size:24px; margin-top:6px;}
.kpi-sub {font-size:12px; color:rgba(0,0,0,0.55); margin-top:6px;}
.kpi-info {font-size:14px; margin-left:8px; color:#666;}
.chart-card {padding:12px; background:#fff; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.04);}
</style>
"""
st.markdown(POWERBI_CSS, unsafe_allow_html=True)

# ----------------------------
# OPENAI client init (if present in secrets)
# ----------------------------
OPENAI_CLIENT = None
if openai_available:
    try:
        # prefer st.secrets for production
        OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
    except Exception:
        OPENAI_CLIENT = None
else:
    OPENAI_CLIENT = None

# ----------------------------
# Helpers (safe operations)
# ----------------------------
def safe_sum(obj: Any) -> int:
    try:
        if isinstance(obj, dict):
            return int(sum(obj.values()))
        if isinstance(obj, pd.Series):
            return int(obj.sum())
        if isinstance(obj, (list, tuple, np.ndarray)):
            return int(sum(obj))
        return int(obj)
    except Exception:
        return 0

def sample_values(series: pd.Series, n=30) -> List[str]:
    s = series.dropna().astype(str)
    if s.empty:
        return []
    if len(s) <= n:
        return s.tolist()
    # sample deterministically for reproducibility
    return s.sample(n, random_state=42).tolist()

# ----------------------------
# Pattern detectors (regex + heuristics)
# ----------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?\d[\d\s\-\(\)]{6,}\d$")
URL_RE = re.compile(r"^https?://")
POSTAL_RE = re.compile(r"^\d{4,5}$")
IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$", re.I)

def detect_patterns_for_series(s: pd.Series) -> Dict[str, float]:
    """Return ratio of various patterns in the series (values are fractions 0..1)."""
    res = {}
    arr = s.dropna().astype(str)
    n = len(arr)
    if n == 0:
        # all zeros to avoid division issues
        for k in ["email","phone","url","postal","iban","date","numeric","alpha","has_digits","likely_name"]:
            res[k] = 0.0
        return res

    # bool masks
    emails = arr.str.match(EMAIL_RE)
    phones = arr.str.match(PHONE_RE)
    urls = arr.str.match(URL_RE)
    postals = arr.str.match(POSTAL_RE)
    ibans = arr.str.match(IBAN_RE)
    # numeric detection: can cast to float
    def is_num(x):
        try:
            float(x)
            return True
        except:
            return False
    numeric_mask = arr.apply(is_num)

    # date detection: try pd.to_datetime
    try:
        dt = pd.to_datetime(arr, errors="coerce", dayfirst=True)
        dates_mask = dt.notna()
    except Exception:
        dates_mask = pd.Series([False]*n, index=arr.index)

    # alpha: pure letters (names)
    alpha_mask = arr.str.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ\-\s']+$")

    # digits present
    has_digits_mask = arr.str.contains(r"\d")

    # likely name heuristic: many tokens capitalized / short words
    def likely_name_val(val: str) -> bool:
        tokens = val.strip().split()
        if not tokens:
            return False
        # require that most tokens start with uppercase letter and have len <= 20
        up = sum(1 for t in tokens if len(t)>0 and t[0].isupper())
        return (up / len(tokens)) >= 0.6 and len(tokens) <= 60

    likely_name_mask = arr.apply(likely_name_val)

    masks = {
        "email": emails,
        "phone": phones,
        "url": urls,
        "postal": postals,
        "iban": ibans,
        "date": dates_mask,
        "numeric": numeric_mask,
        "alpha": alpha_mask,
        "has_digits": has_digits_mask,
        "likely_name": likely_name_mask
    }

    for k, m in masks.items():
        res[k] = float(m.sum()) / float(n)

    return res

# ----------------------------
# Logical type inference
# ----------------------------
LOGICAL_TYPES = [
    "email","phone","date","numeric","id","postal_code","url","iban","name","categorical","text","unknown"
]

def infer_logical_type(patterns: Dict[str,float], series: pd.Series) -> Tuple[str, Dict[str,float]]:
    """
    Decide logical type from detected pattern ratios and heuristics.
    Returns (logical_type, scores/details).
    """
    # simple rules: highest ratio wins among precise patterns
    # Priorities: email, phone, url, iban, postal, date, numeric, name, alpha/categorical, text
    # Use thresholds
    if patterns["email"] >= 0.5:
        return "email", patterns
    if patterns["phone"] >= 0.5:
        return "phone", patterns
    if patterns["url"] >= 0.4:
        return "url", patterns
    if patterns["iban"] >= 0.2:
        return "iban", patterns
    if patterns["postal"] >= 0.6:
        return "postal_code", patterns
    if patterns["date"] >= 0.4:
        return "date", patterns
    if patterns["numeric"] >= 0.6:
        # check id-like (unique high cardinality equal rows)
        n = len(series)
        nunique = series.nunique(dropna=True)
        if nunique / max(1, n) > 0.9:
            return "id", patterns
        return "numeric", patterns
    if patterns["likely_name"] >= 0.5 or patterns["alpha"] >= 0.6:
        return "name", patterns
    # categorical heuristic
    nunique = series.nunique(dropna=True)
    if nunique <= max(1, min(50, int(len(series)*0.05))):
        return "categorical", patterns
    # fallback text
    return "text", patterns

# ----------------------------
# Semantic anomaly scoring
# ----------------------------
def semantic_anomaly_for_column(col_name: str, logical_type: str, patterns: Dict[str,float]) -> Dict[str,Any]:
    """
    Compute an anomaly score and message: 0..1 where 1 means highly anomalous.
    We'll consider:
      - if column name suggests a type that differs from inferred type (heuristic)
      - if inferred type ratio is low (e.g. only 20% emails) -> partial anomaly
    """
    # check name hints
    name_lower = col_name.lower()
    name_hint = "unknown"
    if any(k in name_lower for k in ["mail","email","e-mail","courriel"]):
        name_hint = "email"
    elif any(k in name_lower for k in ["phone","tel","mobile","gsm","telephone","tél"]):
        name_hint = "phone"
    elif "date" in name_lower or "jour" in name_lower or "annee" in name_lower or "year" in name_lower:
        name_hint = "date"
    elif any(k in name_lower for k in ["nom","prenom","name","first","last","surname"]):
        name_hint = "name"
    elif "id" in name_lower or "code" in name_lower or "ref" in name_lower or "num" in name_lower:
        name_hint = "id"
    elif any(k in name_lower for k in ["postal","zip","cp","postcode"]):
        name_hint = "postal_code"

    # strength of inferred type (confidence)
    confidence = 0.0
    # map logical_type to a relevant patterns key
    mapping = {
        "email":"email","phone":"phone","url":"url","iban":"iban",
        "postal_code":"postal","date":"date","numeric":"numeric","id":"numeric",
        "name":"likely_name","categorical":"numeric","text":"alpha"
    }
    key = mapping.get(logical_type, None)
    if key:
        confidence = patterns.get(key, 0.0)
    else:
        # fallback average of numeric/date/email presence
        confidence = max(patterns.get("numeric", 0), patterns.get("date",0), patterns.get("email",0))

    # anomaly due to mismatch with name hint
    mismatch = 0.0
    if name_hint != "unknown" and name_hint != logical_type:
        # severity depends on confidence of inferred type and presence of hint
        mismatch = 0.5 * (1.0 - confidence) + 0.5
    else:
        mismatch = 0.0

    # partial_conformity: if confidence < 0.6 -> partial anomaly
    partial = max(0.0, 0.6 - confidence)  # 0 when confidence >=0.6

    # final anomaly score in [0,1]
    anomaly = min(1.0, mismatch + partial)

    info = {
        "col_name": col_name,
        "name_hint": name_hint,
        "logical_type": logical_type,
        "confidence": round(float(confidence), 3),
        "mismatch_with_name": round(float(mismatch),3),
        "partial_anomaly": round(float(partial),3),
        "anomaly_score": round(float(anomaly),3)
    }
    return info

# ----------------------------
# OpenAI per-column refinement (optional)
# ----------------------------
def openai_refine_column(client: Any, col_name: str, sample_vals: List[str]) -> str:
    """
    Send a compact prompt to OpenAI to get textual judgement on a column sample.
    Returns the raw text answer.
    """
    if client is None:
        return "OpenAI non configuré."
    # build prompt
    sample_text = "\n".join(f"- {v}" for v in sample_vals[:40])
    prompt = f"""
Tu es un expert en data quality. Voici un échantillon de valeurs pour une colonne nommée "{col_name}":
{sample_text}

1) Indique quel est le type logique le plus probable parmi: email, phone, date, name, numeric, id, postal_code, url, iban, city, country, text.
2) Donne un pourcentage estimé d'anomalies (valeurs qui ne correspondent pas au type attendu).
3) Donne 2-3 exemples de règles simples (regex/heuristique) permettant de tester la colonne.

Réponds en texte clair, sans autre contenu.
"""
    # try responses API then fallback
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_tokens=400, temperature=0.1)
        # extract text
        txt = ""
        # resp.output may contain content pieces
        if hasattr(resp, "output") and isinstance(resp.output, list):
            for item in resp.output:
                if isinstance(item, dict) and "content" in item:
                    for c in item["content"]:
                        if isinstance(c, dict) and "text" in c:
                            txt += c["text"]
                elif isinstance(item, str):
                    txt += item
        elif hasattr(resp, "output_text"):
            txt = resp.output_text
        return txt.strip()
    except Exception as e:
        # fallback to older chat completions if available
        try:
            if hasattr(client, "chat"):
                resp2 = client.chat.completions.create(model="gpt-4o-mini",
                                                      messages=[{"role":"system","content":"Tu es un expert en data quality."},
                                                                {"role":"user","content":prompt}],
                                                      max_tokens=400, temperature=0.1)
                return resp2.choices[0].message.content
        except Exception:
            return f"Erreur OpenAI: {e}"
    return "Erreur OpenAI inconnue."

# ----------------------------
# UI / Main flow
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Dashboard", "Contact"])

if page == "Contact":
    st.title("Contact")
    st.markdown("""
**Nom** : SOUMANO Seydou  
**E-mail** : soumanoseydou@icloud.com  
**Téléphone** : +33 6 64 67 88 87  
**LinkedIn** : https://linkedin.com/in/seydou-soumano  
**GitHub** : https://github.com/Ssoumano
""")
    st.stop()

# Main dashboard
st.title("📊 Data Quality — HYBRID (Regex + IA optionnelle)")
st.write("Importe un fichier CSV/XLSX. L'analyse est générique — s'adapte à tout dataset.")

uploaded_file = st.file_uploader("Importer un fichier", type=["csv","xlsx","xls"])
if uploaded_file is None:
    st.info("Téléverse un fichier pour commencer l'analyse.")
    st.stop()

# Load dataset robustly
try:
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        sample = raw[:4096].decode(errors="ignore")
        sep = "," if "," in sample else ";" if ";" in sample else ","
        df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
    else:
        df = pd.read_excel(io.BytesIO(raw))
except Exception as e:
    st.error(f"Impossible de lire le fichier: {e}")
    st.stop()

st.subheader("Aperçu (limité)")
try:
    st.dataframe(df.head(300))
except Exception:
    st.write("Aperçu non disponible (dataset volumineux).")

# Basic profiling (reusing earlier profile_data_quality idea)
@st.cache_data
def profile_basic(df: pd.DataFrame) -> Dict[str,Any]:
    profil = {}
    profil["rows"] = int(df.shape[0])
    profil["cols"] = int(df.shape[1])
    profil["missing_count"] = df.isna().sum().to_dict()
    profil["missing_pct"] = (df.isna().mean()*100).round(2).to_dict()
    profil["dtypes"] = df.dtypes.astype(str).to_dict()
    profil["constant_columns"] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    profil["empty_columns"] = [c for c in df.columns if df[c].dropna().shape[0] == 0]
    profil["duplicate_rows"] = int(df.duplicated().sum())
    # numeric stats if any
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 0:
        try:
            profil["numeric_stats"] = numeric.describe().T.to_dict(orient="index")
        except Exception:
            profil["numeric_stats"] = {}
    else:
        profil["numeric_stats"] = {}
    # outliers
    outliers = {}
    for col in numeric.columns:
        x = numeric[col].dropna()
        if x.empty:
            outliers[col] = 0
            continue
        q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3-q1
        outliers[col] = int(((x < q1 - 1.5*iqr) | (x > q3 + 1.5*iqr)).sum())
    profil["outliers"] = outliers
    # global score simple (keep your earlier weights but will add semantic later)
    miss_mean = np.mean(list((pd.Series(profil["missing_pct"]).fillna(0).values))) if profil["missing_pct"] else 0
    miss_score = max(0, 100 - miss_mean)
    dup_score = max(0, 100 - (profil["duplicate_rows"]/max(1, profil["rows"])) * 100)
    out_values = list(profil["outliers"].values()) if profil.get("outliers") else []
    out_score = max(0, 100 - np.mean(out_values)) if out_values else 100
    profil["global_score_base"] = round((miss_score*0.5 + dup_score*0.3 + out_score*0.2), 1)
    return profil

profil = profile_basic(df)

# Detect logical types and compute semantic anomalies across all columns
col_types = {}
col_patterns = {}
col_semantics = {}
for col in df.columns:
    s = df[col]
    patterns = detect_patterns_for_series(s)
    logical_type, _ = infer_logical_type(patterns, s)
    sem = semantic_anomaly_for_column(col, logical_type, patterns)
    col_types[col] = logical_type
    col_patterns[col] = patterns
    col_semantics[col] = sem

# compute semantic score: average (1 - anomaly) * 100
anomaly_scores = [v["anomaly_score"] for v in col_semantics.values()] if col_semantics else [0]
semantic_score = 100 * (1 - np.mean(anomaly_scores)) if anomaly_scores else 100
semantic_score = round(float(max(0, min(100, semantic_score))), 1)

# combined global score: we can weight semantic as 30% (configurable)
WEIGHT_BASE = 0.7  # base score weight (missing/dup/out)
WEIGHT_SEMANTIC = 0.3
global_score = round(profil["global_score_base"] * WEIGHT_BASE + semantic_score * WEIGHT_SEMANTIC, 1)

# KPI tiles
st.markdown("---")
c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    st.markdown(f"""
    <div class="kpi-tile" title="Score combiné (base + qualité sémantique)">
      <div class="kpi-left">
        <div class="kpi-label">Score global <span class="kpi-info">ℹ️</span></div>
        <div class="kpi-value">{global_score}%</div>
        <div class="kpi-sub">Base + Sémantique</div>
      </div><div style="font-size:36px;">📊</div></div>
    """, unsafe_allow_html=True)
with c2:
    missing_total = safe_sum(profil["missing_count"])
    st.markdown(f"""
    <div class="kpi-tile" title="Nombre total de cellules vides">
      <div class="kpi-left"><div class="kpi-label">Valeurs manquantes <span class="kpi-info">ℹ️</span></div>
      <div class="kpi-value">{missing_total}</div><div class="kpi-sub">Total NA</div></div><div style="font-size:36px;">❗</div></div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-tile" title="Doublons">
      <div class="kpi-left"><div class="kpi-label">Doublons <span class="kpi-info">ℹ️</span></div>
      <div class="kpi-value">{profil['duplicate_rows']}</div><div class="kpi-sub">Lignes identiques</div></div><div style="font-size:36px;">📑</div></div>
    """, unsafe_allow_html=True)
with c4:
    cols_problem = len(profil["empty_columns"]) + len(profil["constant_columns"])
    st.markdown(f"""
    <div class="kpi-tile" title="Colonnes vides ou constantes">
      <div class="kpi-left"><div class="kpi-label">Colonnes vides/constantes <span class="kpi-info">ℹ️</span></div>
      <div class="kpi-value">{cols_problem}</div><div class="kpi-sub">Sans variance</div></div><div style="font-size:36px;">📦</div></div>
    """, unsafe_allow_html=True)

# show semantic score separately
st.markdown("---")
st.write(f"**Score sémantique :** {semantic_score}% — (moyenne d'anomalies colonnes). Poids dans le global : {int(WEIGHT_SEMANTIC*100)}%")
st.info("Le score sémantique mesure si le contenu des colonnes correspond à un type logique attendu (email, date, name...).")

# Display per-column semantic table
st.subheader("Diagnostic sémantique par colonne")
sem_df = pd.DataFrame.from_dict(col_semantics, orient="index")
# add inferred type
sem_df["inferred_type"] = sem_df["logical_type"]
sem_df = sem_df[["inferred_type","confidence","anomaly_score","name_hint","mismatch_with_name","partial_anomaly"]]
sem_df = sem_df.sort_values("anomaly_score", ascending=False)
st.dataframe(sem_df.style.format({"confidence":"{:.2f}","anomaly_score":"{:.3f}"}), use_container_width=True)

# show top examples when high anomaly
st.markdown("### Colonnes avec anomalies élevées (anomaly_score > 0.4)")
high_anom_cols = [c for c,v in col_semantics.items() if v["anomaly_score"]>0.4]
for col in high_anom_cols:
    st.markdown(f"**{col}** — inferred: *{col_semantics[col]['logical_type']}* — anomaly: {col_semantics[col]['anomaly_score']}")
    sample = sample_values(df[col], n=12)
    st.write(sample)

# Column interactive analysis + IA refine
st.markdown("---")
st.subheader("Analyse détaillée d'une colonne (heuristiques + IA optionnelle)")
col_choice = st.selectbox("Choisir une colonne", df.columns.tolist())
if col_choice:
    s = df[col_choice]
    patterns = col_patterns[col_choice]
    inferred_type = col_types[col_choice]
    seminfo = col_semantics[col_choice]

    cA, cB = st.columns([1,2])
    with cA:
        st.write("Inferred type:", inferred_type)
        st.write("Confidence (pattern):", round(patterns.get({
            "email":"email","phone":"phone","url":"url",
            "postal":"postal","iban":"iban","date":"date","numeric":"numeric",
            "alpha":"alpha","likely_name":"likely_name"
        }.get(inferred_type,"numeric"),0),3))
        st.write("Anomaly score:", seminfo["anomaly_score"])
        st.write("Name hint:", seminfo["name_hint"])
    with cB:
        st.write("Top patterns (ratios):")
        st.json({k:round(v,3) for k,v in patterns.items()})

    st.markdown("Exemples de valeurs (sample)")
    st.write(sample_values(s, n=30))

    # Option: refine with OpenAI
    if OPENAI_CLIENT:
        if st.button(f"Générer rapport IA pour la colonne '{col_choice}'"):
            with st.spinner("Appel OpenAI en cours..."):
                try:
                    txt = openai_refine_column(OPENAI_CLIENT, col_choice, sample_values(s, n=40))
                    st.markdown("**Résultat IA :**")
                    st.write(txt)
                except Exception as e:
                    st.error(f"Erreur OpenAI : {e}")
    else:
        st.info("OpenAI non configuré — pas de raffinement IA possible (ajoute OPENAI_API_KEY dans secrets).")

# Visualizations: missing top, outlier heatmap if numeric, types distribution
st.markdown("---")
st.subheader("Visualisations")
# missing bar
mp = pd.Series(profil["missing_pct"])
if mp.sum() > 0:
    top_missing = mp.sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8,4))
    colors = ['#F44336' if v>50 else '#FFC107' if v>20 else '#4CAF50' for v in top_missing.values]
    top_missing.plot(kind='barh', ax=ax, color=colors)
    ax.invert_yaxis(); ax.set_xlabel("% manquant"); ax.set_title("Top colonnes manquantes")
    for i,v in enumerate(top_missing.values):
        ax.text(v+0.5,i,f"{v:.1f}%")
    st.pyplot(fig)
else:
    st.info("Aucune valeur manquante significative détectée.")

# Outliers heatmap
out_map = profil.get("outliers", {})
if out_map:
    df_out = pd.DataFrame.from_dict(out_map, orient="index", columns=["outliers"]).sort_values("outliers", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(2,0.35*len(df_out))))
    sns.heatmap(df_out, annot=True, fmt="d", cmap="Reds", linewidths=.5, ax=ax, cbar_kws={'label':'# outliers'})
    ax.set_ylabel(""); ax.set_title("Outliers par colonne (IQR)")
    st.pyplot(fig)
else:
    st.info("Aucune colonne numérique pour détecter des outliers.")

# types distribution pie
type_counts = pd.Series(list(col_types.values())).value_counts()
fig, ax = plt.subplots(figsize=(6,4))
type_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=['#2196F3','#4CAF50','#FFC107','#F44336','#9C27B0','#00BCD4'])
ax.set_ylabel(""); ax.set_title("Distribution des types logiques détectés")
st.pyplot(fig)

# Quick wins suggestions (basic)
st.markdown("---")
st.subheader("Quick Wins (automatiques)")
suggestions = []
if profil["empty_columns"]:
    suggestions.append(("Supprimer colonnes vides", profil["empty_columns"]))
if profil["constant_columns"]:
    suggestions.append(("Supprimer colonnes constantes", profil["constant_columns"]))
if profil["duplicate_rows"] > 0:
    suggestions.append(("Supprimer doublons", ["all rows"]))

# Add semantic-specific suggestions: columns with high anomaly
for col, sem in col_semantics.items():
    if sem["anomaly_score"] > 0.5:
        suggestions.append((f"Inspecter colonne sémantiquement: {col}", [f"inferred:{sem['logical_type']} - anomaly {sem['anomaly_score']}"]))

if suggestions:
    for title, cols in suggestions:
        st.write(f"- **{title}** — {cols}")
else:
    st.success("Aucune Quick Win automatique détectée.")

# Export HTML report (simple)
st.markdown("---")
st.subheader("Export / Rapport")
def html_report(synthesis_markdown: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_meta = f"<ul><li>Lignes: {profil['rows']:,}</li><li>Colonnes: {profil['cols']}</li><li>Score global: {global_score}%</li><li>Score sémantique: {semantic_score}%</li></ul>"
    # attach small charts as base64 images? keep minimal for now
    html = f"<html><body><h1>Rapport Data Quality</h1><h3>Meta</h3>{rows_meta}<h3>Synthèse</h3><div>{synthesis_markdown.replace('\\n','<br/>')}</div><footer>Généré: {now}</footer></body></html>"
    return html

# produce a small local synthesis fallback (or show button to call IA full synthesis)
local_synth = f"Dataset {profil['rows']:,}×{profil['cols']}. Score global: {global_score}%. Score sémantique: {semantic_score}%. Colonnes top anomalous: {', '.join(high_anom_cols[:5]) if high_anom_cols else 'None'}."

html = html_report(local_synth)
b64 = base64.b64encode(html.encode()).decode()
st.markdown(f'<a href="data:text/html;base64,{b64}" download="report_dq.html">📄 Télécharger rapport HTML</a>', unsafe_allow_html=True)

# Optionally produce PDF if pdfkit installed
if pdfkit_available:
    if st.button("📄 Générer PDF (via wkhtmltopdf)"):
        with st.spinner("Conversion HTML→PDF ..."):
            try:
                pdf_bytes = pdfkit.from_string(html, False)
                st.download_button("📥 Télécharger PDF", data=pdf_bytes, file_name=f"rapport_dq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur conversion PDF: {e}")
else:
    st.info("pdfkit/wkhtmltopdf non présent sur l'environnement — téléchargement HTML disponible.")

st.success("Analyse terminée ✅")
