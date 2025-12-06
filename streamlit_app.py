# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime

st.set_page_config(page_title="Data Quality Platform", layout="wide")

# Try to import helper modules (they were defined in the refactor plan).
# If they are missing, provide safe default implementations.
try:
    import styles
except Exception:
    styles = None

try:
    import utils_data
except Exception:
    utils_data = None

try:
    import utils_openai
except Exception:
    utils_openai = None

try:
    import utils_pdf
except Exception:
    utils_pdf = None


# ----------------------------
# Fallbacks if helper modules missing
# ----------------------------
def fallback_load_dataframe(uploaded_file):
    """Simple fallback loader if utils_data not available."""
    try:
        name = uploaded_file.name.lower()
        data = uploaded_file.getvalue()
        if name.endswith(".csv"):
            # naive separator detection
            sample = data[:4096].decode(errors="ignore")
            sep = "," if "," in sample else ";" if ";" in sample else ","
            return pd.read_csv(io.BytesIO(data), sep=sep, engine="python")
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(data))
        else:
            return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None


def fallback_profile(df: pd.DataFrame) -> dict:
    """Simplified profiling fallback."""
    profil = {}
    profil["rows"] = len(df)
    profil["cols"] = df.shape[1]
    profil["missing_count"] = df.isna().sum()
    profil["missing_pct"] = (df.isna().mean() * 100).round(2)
    profil["dtypes"] = df.dtypes.astype(str)
    profil["constant_columns"] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    profil["empty_columns"] = [c for c in df.columns if df[c].dropna().shape[0] == 0]
    profil["duplicate_rows"] = int(df.duplicated().sum())

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        profil["numeric_stats"] = pd.DataFrame()
        profil["outliers"] = {}
    else:
        profil["numeric_stats"] = numeric.describe().T
        outliers = {}
        for col in numeric.columns:
            s = numeric[col].dropna()
            if s.empty:
                outliers[col] = 0
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers[col] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        profil["outliers"] = outliers

    # global score
    missing_mean = profil["missing_pct"].mean() if hasattr(profil["missing_pct"], "mean") else np.mean(list(profil["missing_pct"]))
    miss_score = max(0, 100 - missing_mean)
    dup_score = max(0, 100 - (profil["duplicate_rows"] / max(1, profil["rows"])) * 100)
    out_score = max(0, 100 - (np.mean(list(profil["outliers"].values())) if profil["outliers"] else 0))
    profil["global_score"] = round(miss_score * 0.5 + dup_score * 0.3 + out_score * 0.2, 1)

    return profil


def fallback_export_pdf_html(synthesis, profil):
    """Fallback simple html-to-pdf link using HTML blob (no pdfkit)."""
    html = f"""
    <html><body>
    <h1>Rapport Data Quality</h1>
    <p><strong>Lignes:</strong> {profil['rows']} &nbsp; <strong>Colonnes:</strong> {profil['cols']}</p>
    <h3>Synthèse</h3>
    <pre>{synthesis}</pre>
    </body></html>
    """
    b = html.encode("utf-8")
    b64 = base64.b64encode(b).decode()
    href = f'data:text/html;base64,{b64}'
    return href  # user can open and save as HTML; PDF generation better with utils_pdf


# ----------------------------
# Helpers for charts (local)
# ----------------------------
def create_missing_chart_local(profil, top_n=10):
    mp = profil["missing_pct"]
    if isinstance(mp, dict):
        mp = pd.Series(mp)
    mp = mp.sort_values(ascending=False).head(top_n)
    if mp.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = ['#F44336' if x > 50 else '#FFC107' if x > 20 else '#4CAF50' for x in mp.values]
    mp.plot(kind='barh', ax=ax, color=colors_list)
    ax.set_xlabel("Pourcentage manquant (%)")
    ax.invert_yaxis()
    for i, v in enumerate(mp.values):
        ax.text(v + 0.8, i, f"{v:.1f}%", va='center')
    plt.tight_layout()
    return fig


def create_types_chart_local(col_types):
    s = pd.Series(col_types).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    s.plot(kind='pie', autopct="%1.1f%%", startangle=90, ax=ax, colors=['#2196F3', '#4CAF50', '#FFC107', '#F44336'])
    ax.set_ylabel("")
    ax.set_title("Répartition des types de colonnes")
    plt.tight_layout()
    return fig


# ----------------------------
# UI
# ----------------------------
if styles is not None and hasattr(styles, "POWERBI_CSS"):
    st.markdown(styles.POWERBI_CSS, unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Testez la qualité de vos données", "Contact"])

# create OpenAI client if utils_openai exists and secret provided
openai_client = None
if utils_openai is not None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            openai_client = utils_openai.get_openai_client(st.secrets["OPENAI_API_KEY"])
        else:
            # try to get client object via environment if utils_openai handles it (optional)
            try:
                openai_client = utils_openai.get_openai_client(None)
            except Exception:
                openai_client = None
    except Exception:
        openai_client = None

# page: tester
if page == "Testez la qualité de vos données":
    st.title("📊 Data Quality Platform — PRO")
    st.write("Importez un fichier (CSV / XLSX). L'app analysera la qualité et proposera synthèses & tests.")

    uploaded_file = st.file_uploader("Importer un fichier", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("Téléverse un fichier pour commencer.")
    else:
        # Load using utils_data if present, else fallback
        if utils_data is not None and hasattr(utils_data, "load_dataframe"):
            try:
                df = utils_data.load_dataframe(uploaded_file)
            except Exception:
                df = fallback_load_dataframe(uploaded_file)
        else:
            df = fallback_load_dataframe(uploaded_file)

        if df is None:
            st.error("Impossible de lire le fichier. Vérifie le format/encodage.")
        else:
            # profiling with utils_data if present
            if utils_data is not None and hasattr(utils_data, "profile_data_quality"):
                try:
                    profil = utils_data.profile_data_quality(df)
                except Exception:
                    profil = fallback_profile(df)
            else:
                profil = fallback_profile(df)

            # basic column type detection (local) - used for UI choices
            col_types = {}
            for col in df.columns:
                if df[col].isna().all():
                    col_types[col] = "empty"
                    continue
                # try datetime
                try:
                    conv = pd.to_datetime(df[col], errors="coerce")
                    if conv.notna().sum() > len(df) * 0.75:
                        col_types[col] = "datetime"
                        continue
                except Exception:
                    pass
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_types[col] = "numeric"
                else:
                    nun = df[col].nunique(dropna=True)
                    col_types[col] = "categorical" if (nun / max(1, len(df)) < 0.05) else "text"

            # KPI tiles (simple)
            c1, c2, c3, c4 = st.columns(4, gap="large")
            c1.metric("Score global", f"{profil['global_score']}%")
            c2.metric("Valeurs manquantes (total)", int(sum(profil["missing_count"].values())) if isinstance(profil["missing_count"], (list, tuple, pd.Series)) or isinstance(profil["missing_count"], dict) else int(profil["missing_count"].sum()))
            c3.metric("Doublons (lignes)", profil["duplicate_rows"])
            c4.metric("Colonnes vides/constantes", len(profil["empty_columns"]) + len(profil["constant_columns"]))

            st.markdown("---")
            st.subheader("Aperçu du DataFrame")
            st.dataframe(df.head(300))

            # Outlier heatmap if numeric columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and any([v > 0 for v in (profil.get("outliers", {}) or {}).values()]):
                st.subheader("Heatmap - Outliers (IQR)")
                outlier_df = pd.DataFrame(list(profil.get("outliers", {}).items()), columns=["col", "outliers"]).set_index("col")
                fig_heat, ax_heat = plt.subplots(figsize=(8, max(2, len(outlier_df)*0.35)))
                sns.heatmap(outlier_df, annot=True, fmt="d", cmap="Reds", ax=ax_heat, cbar_kws={"label": "Nb outliers"})
                st.pyplot(fig_heat)
            else:
                st.info("Aucune colonne numérique contenant des outliers détectée, heatmap non affichée.")

            # types + missing charts
            fig_types = create_types_chart_local(col_types)
            fig_missing = create_missing_chart_local(profil)

            colA, colB = st.columns([1, 1])
            with colA:
                st.subheader("Répartition des types de colonnes")
                st.pyplot(fig_types)
            with colB:
                st.subheader("Top colonnes avec valeurs manquantes")
                if fig_missing is not None:
                    st.pyplot(fig_missing)
                else:
                    st.info("Aucune valeur manquante significative détectée.")

            st.markdown("---")
            st.subheader("Synthèse & Priorités (OpenAI)")
            if openai_client is None:
                st.info("OpenAI non configuré — ajoute OPENAI_API_KEY dans les secrets pour activer les synthèses.")
                # show a small local quick summary
                st.write("Résumé rapide local :")
                st.write(f"- Lignes : {profil['rows']:,}")
                st.write(f"- Colonnes : {profil['cols']}")
                st.write(f"- Score global : {profil['global_score']}%")
            else:
                if st.button("Générer la synthèse IA"):
                    with st.spinner("Appel à OpenAI en cours..."):
                        try:
                            # use utils_openai.generate_synthesis if available
                            if utils_openai is not None and hasattr(utils_openai, "generate_synthesis"):
                                synthesis = utils_openai.generate_synthesis(openai_client, df, profil)
                            else:
                                synthesis = utils_openai.generate_synthesis(openai_client, df, profil) if utils_openai is not None else "Erreur: utils_openai introuvable"
                            st.markdown(synthesis)
                        except Exception as e:
                            st.error(f"Erreur OpenAI: {e}")

            st.markdown("---")
            st.subheader("Tests complémentaires suggérés (OpenAI)")
            if openai_client is None:
                st.info("OpenAI non configuré — impossible de générer les tests IA.")
            else:
                if st.button("Générer les tests IA"):
                    with st.spinner("Génération des tests..."):
                        try:
                            if utils_openai is not None and hasattr(utils_openai, "generate_tests"):
                                tests_text = utils_openai.generate_tests(openai_client, df)
                            else:
                                tests_text = utils_openai.generate_tests(openai_client, df) if utils_openai is not None else "Erreur: utils_openai introuvable"
                            st.markdown(tests_text)
                        except Exception as e:
                            st.error(f"Erreur OpenAI: {e}")

            # per-column profiling interactive
            st.markdown("---")
            st.subheader("Profiling par colonne")
            col_selected = st.selectbox("Choisir une colonne", list(df.columns))
            st.write("Type détecté:", col_types.get(col_selected, "unknown"))
            st.write("Valeurs manquantes:", f"{df[col_selected].isna().sum()} ({df[col_selected].isna().mean()*100:.1f}%)")
            st.write("Valeurs uniques:", df[col_selected].nunique(dropna=True))

            if pd.api.types.is_numeric_dtype(df[col_selected]):
                fig_col, axes = plt.subplots(1, 2, figsize=(12, 4))
                df[col_selected].dropna().hist(bins=30, ax=axes[0], color="#118DFF", edgecolor="white")
                axes[0].set_title("Distribution")
                df[col_selected].dropna().plot.box(ax=axes[1])
                axes[1].set_title("Boxplot")
                st.pyplot(fig_col)
                st.write(df[col_selected].describe().to_frame())
            else:
                st.write(df[col_selected].value_counts().head(20).to_frame())

            # export PDF (use utils_pdf if available)
            st.markdown("---")
            st.subheader("Export / Téléchargement")
            if utils_pdf is not None and hasattr(utils_pdf, "generate_pdf_html") and hasattr(utils_pdf, "export_pdf"):
                if st.button("Générer le rapport PDF (utils_pdf)"):
                    with st.spinner("Construction PDF (utils_pdf)..."):
                        try:
                            # synth: if not generated earlier, ask utils_openai for a small local summary as fallback
                            synthesis_text = "Synthèse non générée (appuyez sur 'Générer la synthèse IA' pour avoir une synthèse complète)." 
                            # if user generated earlier and 'synthesis' in locals, use it
                            if "synthesis" in locals():
                                synthesis_text = synthesis
                            html_content = utils_pdf.generate_pdf_html(synthesis_text, profil)
                            href = utils_pdf.export_pdf(html_content)
                            st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erreur utils_pdf: {e}")
            else:
                # fallback: build small HTML report and offer as download of HTML
                if st.button("Générer rapport HTML (fallback)"):
                    synthesis_text = locals().get("synthesis", "Synthèse non générée.")
                    href = fallback_export_pdf_html(synthesis_text, profil)
                    st.markdown(f"Ouvrir ce lien, puis enregistrer la page comme PDF/HTML: {href}")

# Contact page
elif page == "Contact":
    st.title("Contact")
    st.write("**Nom :** SOUMANO Seydou")
    st.write("**E-mail :** soumanoseydou@icloud.com")
    st.write("**Téléphone :** +33 6 64 67 88 87")
    st.write("**LinkedIn :** https://linkedin.com/in/seydou-soumano")
    st.write("**GitHub :** https://github.com/Ssoumano")
