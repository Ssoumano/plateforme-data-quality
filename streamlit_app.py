# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import json
import textwrap

from openai import OpenAI

# -------------------------
# Configuration OpenAI
# -------------------------
# Option (A) : clé dans st.secrets (recommandé for deploy)
# Option (B) : clé hardcodée (DEV / TEST only) - remplace si tu veux
DEFAULT_KEY_IN_CODE = "YOUR_API_KEY_HERE"  # <-- remplace par ta clé pour test local uniquement

def get_openai_client():
    key = None
    # 1) streamlit secrets (best for deployment)
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # 2) default in code fallback (development)
    if not key or key == "":
        key = DEFAULT_KEY_IN_CODE if DEFAULT_KEY_IN_CODE != "YOUR_API_KEY_HERE" else None
    if not key:
        return None
    return OpenAI(api_key=key)

client = get_openai_client()

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
    profil['numeric_stats'] = numeric.describe().T if not numeric.empty else pd.DataFrame()

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

    # Score global (simple composite)
    miss_score = max(0, 100 - profil["missing_pct"].mean()) if len(profil["missing_pct"])>0 else 100
    dup_score = max(0, 100 - (profil["duplicate_rows"] / max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(outliers.values())) if outliers else 0))

    profil["global_score"] = round((miss_score*0.5 + dup_score*0.3 + out_score*0.2), 1)

    return profil

def generate_local_report(profil: dict) -> str:
    lines = []
    lines.append("===== RAPPORT DATA QUALITY =====\n")
    lines.append(f"Lignes : {profil['rows']}  | Colonnes : {profil['cols']}\n")
    lines.append(f"Colonnes constantes : {profil['constant_columns']}")
    lines.append(f"Colonnes vides : {profil['empty_columns']}")
    lines.append(f"Doublons : {profil['duplicate_rows']}\n")

    lines.append("\n-- Valeurs manquantes --")
    for col in profil['missing_count'].index:
        lines.append(f"{col}: {profil['missing_count'][col]}  ({profil['missing_pct'][col]}%)")

    lines.append("\n-- Types de colonnes --")
    for col in profil['dtypes'].index:
        lines.append(f"{col}: {profil['dtypes'][col]}")

    lines.append("\n-- Outliers --")
    for col in profil['outliers']:
        lines.append(f"{col}: {profil['outliers'][col]}")

    lines.append("\nRecommandations :")
    lines.append("- Vérifier les colonnes à forte proportion de valeurs manquantes")
    lines.append("- Analyser les doublons")
    lines.append("- Corriger ou supprimer les colonnes vides/constantes")
    lines.append("- Vérifier les types de données incohérents")
    return "\n".join(lines)

# -------------------------
# OpenAI helpers: synthesis + priorities
# -------------------------
def openai_generate_synthesis_and_priorities(df_sample: dict, profil: dict, detailed=True):
    """
    Calls OpenAI to generate:
      - a detailed synthesis (10-15 lines)
      - a JSON array of priorities [{priority, problem, columns, recommendation}, ...]
    Returns: (synthesis_text, priorities_list_or_none, raw_ai_text)
    """
    if client is None:
        return ("OpenAI non configuré. Configure ta clé via st.secrets['OPENAI_API_KEY'] or DEFAULT_KEY_IN_CODE.", None, None)

    # Build prompt: include profil and column samples. Ask for JSON for priorities.
    prompt = textwrap.dedent(f"""
    Tu es un expert en qualité des données (data quality engineer).
    J'ai analysé un dataset et je te fournis :
    - Un petit profil résumé (nombre de lignes/colonnes, % missing par colonne, nb doublons, outliers par colonne)
    - Un échantillon d'exemples pour chaque colonne (maximum 10 valeurs par colonne)

    PROFIL :
    {json.dumps({k: (v if not isinstance(v, (pd.Series, pd.DataFrame)) else v.to_dict()) for k,v in profil.items()}, ensure_ascii=False, indent=2)}

    ECHANTILLON (par colonne) :
    {json.dumps(df_sample, ensure_ascii=False, indent=2)}

    Tâche :
    1) Rédige une synthèse détaillée en français d'environ 10 à 15 lignes expliquant l'état global de la qualité des données, les risques majeurs et les impacts business potentiels. Sois clair, professionnel, et orienté action.
    2) Génére une liste de priorités (High, Medium, Low) sous format JSON strict, exemple :
    [
      {{
        "priority": "High",
        "problem": "Valeurs manquantes élevées",
        "columns": ["date_de_naissance", "date_mandat"],
        "recommendation": "Imputer/valider à la source; vérifier process d'ingestion"
      }},
      ...
    ]
    3) Fournis aussi 3 actions concrètes à court terme (30 jours).

    **Règles de sortie :**
    - PREMIÈRE PARTIE : la synthèse (texte, environ 10-15 lignes)
    - DEUXIÈME PARTIE : la JSON list (strict JSON) correspondant aux priorités
    - TROISIÈME PARTIE : 3 actions courtes séparées par des tirets

    Ne réponds qu'avec ces éléments (synthèse, puis JSON, puis actions), et rien d'autre.
    """)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un expert en data quality, rédaction professionnelle en français."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        ai_text = response.choices[0].message.content

        # Try to parse: assume synthese text, then JSON list, then actions.
        # We'll search for the first occurrence of '[' that starts the JSON priorities.
        json_start = ai_text.find('[')
        json_end = ai_text.find(']')  # first closing bracket
        # but better find matching closing bracket for the first '['
        priorities = None
        synth = ai_text
        actions = None
        if json_start != -1:
            # find matching closing bracket index
            stack = 0
            end_idx = -1
            for i in range(json_start, len(ai_text)):
                if ai_text[i] == '[':
                    stack += 1
                elif ai_text[i] == ']':
                    stack -= 1
                    if stack == 0:
                        end_idx = i
                        break
            if end_idx != -1:
                json_text = ai_text[json_start:end_idx+1]
                try:
                    priorities = json.loads(json_text)
                except Exception:
                    priorities = None
                # synth is text before json_start
                synth = ai_text[:json_start].strip()
                actions = ai_text[end_idx+1:].strip()
        return (synth, priorities, actions)
    except Exception as e:
        return (f"Erreur OpenAI : {e}", None, None)

# -------------------------
# OpenAI: suggest tests (simple)
# -------------------------
def openai_suggest_tests(df):
    if client is None:
        return "OpenAI non configuré."
    sample = {col: df[col].dropna().astype(str).head(10).tolist() for col in df.columns}
    prompt = textwrap.dedent(f"""
    Tu es expert en data quality. Voici un échantillon par colonne :
    {json.dumps(sample, ensure_ascii=False, indent=2)}

    Propose 6 tests pertinents adaptés aux colonnes (format: TEST | COLONNES | DESCRIPTION).
    """)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Expert data quality."},
                      {"role":"user","content":prompt}],
            max_tokens=700,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur OpenAI : {e}"

# -------------------------
# UI / Streamlit
# -------------------------
st.set_page_config(page_title="Data Quality App - Synthèse OpenAI", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Testez la qualité de vos données", "Contact"])

# Info icon helper
def info_icon(txt):
    st.markdown(f"<span style='color:#888; font-size:14px; cursor:pointer;' title='{txt}'>ℹ️</span>", unsafe_allow_html=True)

# Page: Data Quality
if page == "Testez la qualité de vos données":
    st.title("📊 Data Quality Platform")
    st.write("Importe un fichier CSV/Excel — une synthèse détaillée et des priorités seront proposées via OpenAI.")

    uploaded_file = st.file_uploader("📥 Importer un fichier", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("Upload un fichier pour commencer l'analyse.")
    else:
        df = load_dataframe(uploaded_file)
        if df is None:
            st.error("Impossible de charger le fichier.")
        else:
            profil = profile_data_quality(df)

            # KPI row
            st.markdown("## ⭐ Indicateurs clés")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<b>Score global</b> {info_icon('Score composite: missing(50%), doublons(30%), outliers(20%)')}", unsafe_allow_html=True)
            c1.metric("", f"{profil['global_score']}%")
            c2.markdown(f"<b>Valeurs manquantes</b> {info_icon('Somme des valeurs manquantes (NA)')}", unsafe_allow_html=True)
            c2.metric("", int(profil['missing_count'].sum()))
            c3.markdown(f"<b>Doublons</b> {info_icon('Nombre de lignes strictement dupliquées')}", unsafe_allow_html=True)
            c3.metric("", profil['duplicate_rows'])
            c4.markdown(f"<b>Colonnes vides/constantes</b> {info_icon('Colonnes sans variance ou entièrement vides')}", unsafe_allow_html=True)
            c4.metric("", len(profil['empty_columns']) + len(profil['constant_columns']))

            # Data preview
            st.subheader("👀 Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # Outliers heatmap (pro)
            st.subheader("⚠️ Heatmap – Nombre d’outliers (IQR)")
            outliers_df = pd.DataFrame.from_dict(profil['outliers'], orient='index', columns=['outliers']).sort_values('outliers', ascending=False)
            if outliers_df.empty:
                st.info("Aucune colonne numérique détectée pour calculer des outliers.")
            else:
                fig, ax = plt.subplots(figsize=(8, max(2, len(outliers_df)*0.45)))
                sns.heatmap(
                    outliers_df,
                    annot=True,
                    fmt='d',
                    cmap=sns.color_palette("Reds", as_cmap=True),
                    linewidths=.6,
                    linecolor="white",
                    cbar_kws={"label": "Nombre d'outliers"},
                    ax=ax
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("Outliers détectés par colonne (Méthode IQR)", fontsize=14, pad=12)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
                st.pyplot(fig)

            # Numeric stats
            st.subheader("📈 Statistiques numériques")
            if isinstance(profil['numeric_stats'], pd.DataFrame) and not profil['numeric_stats'].empty:
                st.dataframe(profil['numeric_stats'])
            else:
                st.info("Aucune colonne numérique pour afficher des statistiques.")

            # Local textual report
            st.subheader("📝 Rapport automatique (local)")
            info_icon("Rapport produit localement sans IA : valeurs manquantes / types / outliers / recommandations générales")
            st.code(generate_local_report(profil))

            # OpenAI suggestions + synthesis
            st.subheader("🤖 OpenAI — Synthèse détaillée & Priorités")
            if client is None:
                st.error("OpenAI non configuré. Ajoute ta clé dans Streamlit Secrets OPENAI_API_KEY ou dans DEFAULT_KEY_IN_CODE.")
            else:
                do_ai = st.button("Générer synthèse détaillée & priorités (OpenAI)")
                if do_ai:
                    # prepare sample per column (max 10 values each)
                    sample = {col: df[col].dropna().astype(str).head(10).tolist() for col in df.columns}
                    with st.spinner("Génération OpenAI en cours..."):
                        synth, priorities, actions = openai_generate_synthesis_and_priorities(sample, profil, detailed=True)
                    # show synthesis
                    st.markdown("### 🧾 Synthèse détaillée (OpenAI)")
                    if synth:
                        st.write(synth)
                    else:
                        st.info("Aucune synthèse renvoyée par l'IA.")

                    # show priorities as table if JSON parsed
                    st.markdown("### 🔎 Priorités (High / Medium / Low)")
                    if priorities:
                        try:
                            pri_df = pd.DataFrame(priorities)
                            # reorder columns if present
                            cols = [c for c in ["priority","problem","columns","recommendation"] if c in pri_df.columns]
                            st.dataframe(pri_df[cols])
                        except Exception:
                            st.write("Impossible d'afficher le JSON des priorités — voici le texte brut :")
                            st.text(priorities)
                    else:
                        st.info("L'IA n'a pas renvoyé de JSON structuré. Voir texte brut ci-dessous.")
                        st.text(priorities if priorities else "Aucune priorité structurée fournie.")

                    # actions
                    st.markdown("### ✅ Actions concrètes à court terme (30 jours)")
                    if actions:
                        # pretty-print actions (could be lines)
                        lines = [ln.strip("-* \n") for ln in actions.splitlines() if ln.strip()]
                        for ln in lines:
                            st.write(f"- {ln}")
                    else:
                        st.info("Aucune action courte fournie par l'IA.")

                # Also keep the previous "suggested tests" box
                st.subheader("🧪 Tests complémentaires suggérés (OpenAI)")
                if st.button("Demander suggestions de tests (OpenAI)"):
                    with st.spinner("OpenAI propose des tests..."):
                        suggestions = openai_suggest_tests(df)
                    st.text(suggestions)

# Page Contact
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**Github :** https://github.com/Ssoumano")
