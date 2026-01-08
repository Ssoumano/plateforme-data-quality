# styles.py

POWERBI_CSS = """
<style>
/* ==================== LAYOUT & SPACING ==================== */
section.main > div.block-container {
    max-width: 1400px !important;
    padding-top: 2rem;
}

/* Meilleur espacement entre les sections */
.element-container {
    margin-bottom: 1rem;
}

/* ==================== TYPOGRAPHIE ==================== */
h1, h2, h3 {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-weight: 600;
    color: #1e1e1e;
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 1.5rem;
}

h2 {
    font-size: 1.8rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

h3 {
    font-size: 1.3rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

/* ==================== KPI TILES ==================== */
.kpi-tile {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px 24px;
    border-radius: 16px;
    color: white;
    width: 100%;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: none;
}

.kpi-tile:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.35);
}

.kpi-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.9;
    font-weight: 500;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0;
    line-height: 1;
}

.kpi-subtitle {
    font-size: 12px;
    margin-top: 8px;
    opacity: 0.85;
    font-weight: 400;
}

/* Variantes de couleurs pour les KPI */
.kpi-tile.success {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.kpi-tile.warning {
    background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
}

.kpi-tile.danger {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
}

.kpi-tile.info {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

/* ==================== BOUTONS ==================== */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Bouton secondaire */
.stButton.secondary > button {
    background: white;
    color: #667eea;
    border: 2px solid #667eea;
}

/* ==================== MESSAGES INFO/WARNING/ERROR ==================== */
.stAlert {
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ==================== UPLOAD ZONE ==================== */
.stFileUploader {
    border: 2px dashed #667eea;
    border-radius: 16px;
    padding: 32px;
    background: #f8f9ff;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #764ba2;
    background: #f0f2ff;
}

/* ==================== DATAFRAME ==================== */
.dataframe {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.dataframe thead th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600;
    padding: 12px;
    text-align: left;
}

.dataframe tbody tr:nth-child(even) {
    background: #f8f9ff;
}

.dataframe tbody tr:hover {
    background: #eef0ff;
    transition: background 0.2s ease;
}

/* ==================== SELECTBOX ==================== */
.stSelectbox > div > div {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    transition: border-color 0.2s ease;
}

.stSelectbox > div > div:hover {
    border-color: #667eea;
}

/* ==================== SPINNER / LOADING ==================== */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* ==================== TABS ==================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px 12px 0 0;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* ==================== METRICS STREAMLIT ==================== */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
    color: #1e1e1e;
}

[data-testid="stMetricLabel"] {
    font-size: 14px;
    font-weight: 500;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ==================== SIDEBAR ==================== */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
}

.css-1d391kg .stRadio > label {
    font-weight: 500;
    font-size: 15px;
    padding: 8px 12px;
    border-radius: 8px;
    transition: background 0.2s ease;
}

.css-1d391kg .stRadio > label:hover {
    background: #eef0ff;
}

/* ==================== RESPONSIVE ==================== */
@media (max-width: 768px) {
    section.main > div.block-container {
        padding: 1rem;
    }
    
    h1 {
        font-size: 1.8rem;
    }
    
    .kpi-value {
        font-size: 24px;
    }
}

/* ==================== ANIMATIONS ==================== */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.fade-in {
    animation: fadeIn 0.5s ease;
}

/* ==================== TOOLTIPS ==================== */
.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 8px;
    padding: 8px 12px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -60px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* ==================== CARDS ==================== */
.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 20px;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

/* ==================== PROGRESS BAR ==================== */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 8px;
}

/* ==================== DIVIDER ==================== */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, transparent);
    margin: 32px 0;
}
</style>
"""
