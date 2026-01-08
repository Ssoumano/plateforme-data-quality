# utils_pdf.py

import base64
from typing import Dict, Optional
from datetime import datetime
import pandas as pd


def generate_pdf_html(synthesis: str, profil: Dict, df: Optional[pd.DataFrame] = None) -> str:
    """
    Génère un rapport HTML professionnel de la qualité des données.
    
    Args:
        synthesis: Texte de synthèse généré par l'IA
        profil: Dictionnaire de profil de qualité
        df: DataFrame original (optionnel)
        
    Returns:
        str: Code HTML complet du rapport
    """
    
    # Date du rapport
    report_date = datetime.now().strftime("%d/%m/%Y à %H:%M")
    
    # Score global avec couleur
    score = profil['global_score']
    if score >= 80:
        score_color = "#4CAF50"
        score_label = "Excellent"
    elif score >= 60:
        score_color = "#FFC107"
        score_label = "Bon"
    elif score >= 40:
        score_color = "#FF9800"
        score_label = "Moyen"
    else:
        score_color = "#F44336"
        score_label = "Faible"
    
    # Tableau des KPIs principaux
    kpis_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Lignes</div>
            <div class="kpi-value">{profil['rows']:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Colonnes</div>
            <div class="kpi-value">{profil['cols']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Score Global</div>
            <div class="kpi-value" style="color: {score_color};">{score}%</div>
            <div class="kpi-subtitle">{score_label}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Valeurs Manquantes</div>
            <div class="kpi-value">{profil['total_missing']:,}</div>
            <div class="kpi-subtitle">{profil['missing_rate']:.2f}% du dataset</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Lignes Dupliquées</div>
            <div class="kpi-value">{profil['duplicate_rows']:,}</div>
            <div class="kpi-subtitle">{profil['duplicate_rate']:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Outliers</div>
            <div class="kpi-value">{profil.get('total_outliers', 0):,}</div>
        </div>
    </div>
    """
    
    # Scores détaillés
    score_details = profil.get('score_details', {})
    scores_html = f"""
    <div class="scores-section">
        <h3>📊 Scores Détaillés</h3>
        <div class="score-bars">
            <div class="score-item">
                <span class="score-name">Complétude (valeurs manquantes)</span>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: {score_details.get('missing_score', 0)}%; background: linear-gradient(90deg, #667eea, #764ba2);"></div>
                </div>
                <span class="score-percent">{score_details.get('missing_score', 0):.1f}%</span>
            </div>
            <div class="score-item">
                <span class="score-name">Unicité (doublons)</span>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: {score_details.get('duplicate_score', 0)}%; background: linear-gradient(90deg, #11998e, #38ef7d);"></div>
                </div>
                <span class="score-percent">{score_details.get('duplicate_score', 0):.1f}%</span>
            </div>
            <div class="score-item">
                <span class="score-name">Exactitude (outliers)</span>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: {score_details.get('outlier_score', 0)}%; background: linear-gradient(90deg, #f2994a, #f2c94c);"></div>
                </div>
                <span class="score-percent">{score_details.get('outlier_score', 0):.1f}%</span>
            </div>
            <div class="score-item">
                <span class="score-name">Cohérence (colonnes problématiques)</span>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: {score_details.get('problematic_columns_score', 0)}%; background: linear-gradient(90deg, #4facfe, #00f2fe);"></div>
                </div>
                <span class="score-percent">{score_details.get('problematic_columns_score', 0):.1f}%</span>
            </div>
        </div>
    </div>
    """
    
    # Problèmes identifiés
    problems_html = "<ul class='problems-list'>"
    
    if profil['empty_columns']:
        problems_html += f"<li><strong>❌ Colonnes vides:</strong> {', '.join(profil['empty_columns'][:5])}"
        if len(profil['empty_columns']) > 5:
            problems_html += f" <em>(+{len(profil['empty_columns'])-5} autres)</em>"
        problems_html += "</li>"
    
    if profil['constant_columns']:
        problems_html += f"<li><strong>⚠️ Colonnes constantes:</strong> {', '.join(profil['constant_columns'][:5])}"
        if len(profil['constant_columns']) > 5:
            problems_html += f" <em>(+{len(profil['constant_columns'])-5} autres)</em>"
        problems_html += "</li>"
    
    if profil['duplicate_rows'] > 0:
        problems_html += f"<li><strong>🔄 Lignes dupliquées:</strong> {profil['duplicate_rows']:,} ({profil['duplicate_rate']:.2f}%)</li>"
    
    high_missing = profil['missing_pct'][profil['missing_pct'] > 50]
    if not high_missing.empty:
        problems_html += f"<li><strong>📊 Colonnes avec >50% de valeurs manquantes:</strong> {', '.join(high_missing.index.tolist()[:5])}"
        if len(high_missing) > 5:
            problems_html += f" <em>(+{len(high_missing)-5} autres)</em>"
        problems_html += "</li>"
    
    if not problems_html.endswith("</li>"):
        problems_html += "<li><strong>✅ Aucun problème majeur détecté</strong></li>"
    
    problems_html += "</ul>"
    
    # Recommandations
    recommendations_html = "<ul class='recommendations-list'>"
    for rec in profil.get('recommendations', []):
        recommendations_html += f"<li>{rec}</li>"
    recommendations_html += "</ul>"
    
    # Distribution des types (si disponible)
    type_dist = profil.get('type_distribution', {})
    type_dist_html = ""
    if type_dist:
        type_dist_html = "<div class='type-distribution'><h4>Distribution des Types de Colonnes</h4><ul>"
        for type_name, count in type_dist.items():
            pct = (count / profil['cols']) * 100
            type_dist_html += f"<li><strong>{type_name}:</strong> {count} colonnes ({pct:.1f}%)</li>"
        type_dist_html += "</ul></div>"
    
    # HTML complet
    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport Data Quality - {report_date}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.1);
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }}
            
            .header h1 {{
                color: #667eea;
                font-size: 2.5rem;
                margin-bottom: 10px;
            }}
            
            .header .date {{
                color: #666;
                font-size: 0.9rem;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .section h2 {{
                color: #333;
                font-size: 1.8rem;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #eee;
            }}
            
            .section h3 {{
                color: #555;
                font-size: 1.4rem;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .kpi-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 12px;
                color: white;
                text-align: center;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }}
            
            .kpi-label {{
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                opacity: 0.9;
                margin-bottom: 8px;
            }}
            
            .kpi-value {{
                font-size: 2rem;
                font-weight: 700;
                margin: 5px 0;
            }}
            
            .kpi-subtitle {{
                font-size: 0.8rem;
                opacity: 0.85;
                margin-top: 5px;
            }}
            
            .scores-section {{
                background: #f8f9ff;
                padding: 25px;
                border-radius: 12px;
                margin: 30px 0;
            }}
            
            .score-bars {{
                margin-top: 20px;
            }}
            
            .score-item {{
                display: flex;
                align-items: center;
                margin-bottom: 20px;
                gap: 15px;
            }}
            
            .score-name {{
                flex: 0 0 250px;
                font-weight: 500;
                color: #555;
            }}
            
            .score-bar-container {{
                flex: 1;
                height: 24px;
                background: #e0e0e0;
                border-radius: 12px;
                overflow: hidden;
            }}
            
            .score-bar {{
                height: 100%;
                border-radius: 12px;
                transition: width 0.3s ease;
            }}
            
            .score-percent {{
                flex: 0 0 60px;
                text-align: right;
                font-weight: 600;
                color: #667eea;
            }}
            
            .problems-list, .recommendations-list {{
                list-style: none;
                padding: 0;
            }}
            
            .problems-list li, .recommendations-list li {{
                padding: 12px 15px;
                margin-bottom: 10px;
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                border-radius: 6px;
            }}
            
            .recommendations-list li {{
                background: #d1ecf1;
                border-left-color: #0dcaf0;
            }}
            
            .synthesis-box {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                border-left: 5px solid #667eea;
                margin: 20px 0;
                line-height: 1.8;
            }}
            
            .type-distribution {{
                background: #e7f3ff;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            
            .type-distribution ul {{
                list-style: none;
                padding-left: 0;
            }}
            
            .type-distribution li {{
                padding: 8px 0;
                border-bottom: 1px solid #d0e7ff;
            }}
            
            .footer {{
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #eee;
                text-align: center;
                color: #666;
                font-size: 0.9rem;
            }}
            
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                .container {{
                    box-shadow: none;
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Rapport Data Quality</h1>
                <p class="date">Généré le {report_date}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Indicateurs Clés</h2>
                {kpis_html}
            </div>
            
            {scores_html}
            
            <div class="section">
                <h2>⚠️ Problèmes Identifiés</h2>
                {problems_html}
            </div>
            
            <div class="section">
                <h2>💡 Recommandations</h2>
                {recommendations_html}
            </div>
            
            {type_dist_html}
            
            <div class="section">
                <h2>📝 Synthèse Détaillée</h2>
                <div class="synthesis-box">
                    {synthesis.replace(chr(10), '<br>')}
                </div>
            </div>
            
            <div class="footer">
                <p>Rapport généré par <strong>Data Quality Platform - PRO</strong></p>
                <p>Pour plus d'informations: <a href="mailto:soumanoseydou@icloud.com">soumanoseydou@icloud.com</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def export_pdf(html_content: str) -> str:
    """
    Convertit du contenu HTML en lien de téléchargement.
    
    Note: Cette fonction retourne un lien HTML encodé en base64.
    Pour une vraie conversion PDF, il faudrait utiliser pdfkit ou weasyprint.
    
    Args:
        html_content: Contenu HTML à exporter
        
    Returns:
        str: Lien HTML de téléchargement
    """
    try:
        # Tentative d'import de pdfkit pour vraie conversion PDF
        import pdfkit
        
        try:
            pdf_bytes = pdfkit.from_string(html_content, False)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_data_quality.pdf" class="download-btn">📄 Télécharger le rapport PDF</a>'
            return href
        except Exception as e:
            # Fallback si pdfkit ne fonctionne pas
            print(f"Erreur pdfkit: {e}")
            return export_html_fallback(html_content)
            
    except ImportError:
        # Si pdfkit n'est pas installé, utiliser le fallback HTML
        return export_html_fallback(html_content)


def export_html_fallback(html_content: str) -> str:
    """
    Fallback: Export en HTML pur (peut être sauvegardé comme PDF via le navigateur).
    
    Args:
        html_content: Contenu HTML
        
    Returns:
        str: Lien de téléchargement HTML
    """
    b64 = base64.b64encode(html_content.encode('utf-8')).decode()
    
    href = f'''
    <a href="data:text/html;base64,{b64}" 
       download="rapport_data_quality.html" 
       style="
           display: inline-block;
           padding: 12px 24px;
           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           color: white;
           text-decoration: none;
           border-radius: 8px;
           font-weight: 600;
           box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
           transition: transform 0.2s ease;
       "
       onmouseover="this.style.transform='translateY(-2px)'"
       onmouseout="this.style.transform='translateY(0)'">
        📄 Télécharger le rapport HTML
    </a>
    <p style="margin-top: 10px; color: #666; font-size: 0.9rem;">
        💡 Conseil: Vous pouvez ouvrir le fichier HTML et l'enregistrer en PDF via votre navigateur (Imprimer → Enregistrer au format PDF)
    </p>
    '''
    
    return href
