# utils_pdf.py

import pdfkit
import base64

def generate_pdf_html(synthesis, profil):
    html = f"""
    <h1>Rapport Data Quality</h1>

    <h3>Indicateurs clés</h3>
    <ul>
        <li>Lignes : {profil['rows']}</li>
        <li>Colonnes : {profil['cols']}</li>
        <li>Score global : {profil['global_score']}%</li>
        <li>Valeurs manquantes : {int(profil['missing_count'].sum())}</li>
        <li>Doublons : {profil['duplicate_rows']}</li>
    </ul>

    <h3>Synthèse</h3>
    <p>{synthesis}</p>
    """

    return html


def export_pdf(html_content):
    pdf = pdfkit.from_string(html_content, False)
    b64 = base64.b64encode(pdf).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📄 Télécharger le PDF</a>'
