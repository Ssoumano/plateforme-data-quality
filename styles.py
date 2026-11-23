# styles.py
"""CSS léger de l'application.

Les couleurs de marque viennent de `.streamlit/config.toml` (thème Streamlit).
Ici : uniquement de la mise en page et des teintes neutres (rgba) qui restent
lisibles en thème clair comme en thème sombre. Pas de couleur de texte codée en dur.
"""

APP_CSS = """
<style>
/* ---------- Mise en page ---------- */
[data-testid="stMainBlockContainer"], section.main > div.block-container {
    max-width: 1280px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* ---------- Typographie ---------- */
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.02em; padding-bottom: .25rem; }
h2 { font-size: 1.5rem !important; font-weight: 650 !important; }
h3 { font-size: 1.2rem !important; font-weight: 650 !important; padding-top: .5rem !important; }
h4 { font-size: 1.05rem !important; font-weight: 600 !important; }

/* ---------- Métriques ---------- */
[data-testid="stMetricLabel"] {
    font-size: .75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
    opacity: .7;
}
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; }

/* ---------- Onglets : soulignement, défilement horizontal si trop larges ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { padding: .6rem 1rem; font-weight: 600; white-space: nowrap; }

/* ---------- Composants ---------- */
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border-style: dashed; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stExpander"] { border-radius: 10px; }

hr { margin: 1.5rem 0 !important; opacity: .5; }

/* ---------- Cartes (page d'accueil) ---------- */
.card {
    background: rgba(128, 128, 128, .08);
    border: 1px solid rgba(128, 128, 128, .22);
    border-radius: 14px;
    padding: 1.25rem 1rem;
    min-height: 9.5rem;
}
.card h2 { margin: 0 0 .25rem 0; padding: 0; }
.card h4 { margin: 0 0 .25rem 0; }
.card p { margin: 0; opacity: .75; font-size: .95rem; }

/* ---------- Bandeau (page contact) ---------- */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
}
.hero h2, .hero p { color: #fff !important; }

/* ---------- Pied de page ---------- */
.app-footer { text-align: center; opacity: .65; padding: 1rem 0; font-size: .9rem; }

/* ---------- Mobile ---------- */
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"], section.main > div.block-container {
        padding: 1rem 1rem 2rem 1rem;
    }
    h1 { font-size: 1.5rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
}
</style>
"""

POWERBI_CSS = APP_CSS  # alias rétro-compatible
