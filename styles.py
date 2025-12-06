# styles.py

POWERBI_CSS = """
<style>
/* Largeur max */
section.main > div.block-container {
    max-width: 1400px !important;
}

/* Titres */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

/* KPI tiles */
.kpi-tile {
    background-color: var(--bg);
    padding: 16px 20px;
    border-radius: 12px;
    color: var(--text);
    width: 100%;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
}
.kpi-label {
    font-size: 14px;
    color: var(--muted);
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
}
.kpi-subtitle {
    font-size: 12px;
    margin-top: 4px;
    opacity: 0.8;
}
</style>
"""
