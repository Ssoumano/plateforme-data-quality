# Plateforme Data Quality

Application **Streamlit** qui analyse la qualité de n'importe quelle base structurée (CSV / Excel) et produit un rapport détaillé avec des axes d'amélioration, enrichi par l'API OpenAI.

## Fonctionnalités
- Import CSV (séparateur et encodage détectés) et Excel (`.xlsx`, `.xls`)
- Profiling : valeurs manquantes, doublons, colonnes vides/constantes, haute cardinalité, outliers (IQR), corrélations
- Score global de qualité (voir ci-dessous) et « quick wins » actionnables
- Synthèse, tests de qualité et script de nettoyage générés par IA (optionnel)
- Tableau de synthèse par colonne (statut, alertes, filtres) et graphiques interactifs Plotly
- Onglet « Lignes à corriger » : doublons, valeurs manquantes et outliers, téléchargeables en CSV
- Nettoyage configurable avec aperçu avant/après (score, lignes, colonnes)
- Exports : rapport PDF, rapport HTML, CSV nettoyé

## Score global (0-100)
| Composante | Poids | Calcul |
|---|---|---|
| Valeurs manquantes | 40 % | 100 − taux de cellules manquantes |
| Doublons | 25 % | 100 − taux de lignes dupliquées |
| Outliers | 20 % | 100 − part de cellules numériques aberrantes (IQR) |
| Colonnes problématiques | 15 % | 100 − part de colonnes vides ou constantes |

## Installation
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # puis renseigner OPENAI_API_KEY
streamlit run streamlit_app.py
```
Sans clé OpenAI, l'application fonctionne : seules les fonctions IA sont désactivées.

## Confidentialité
Lors d'un appel IA, seules des statistiques agrégées (noms de colonnes, types, taux de manquants) sont envoyées à OpenAI. L'envoi d'exemples de valeurs est **désactivé par défaut** et s'active via une case à cocher dans la barre latérale. À éviter pour des données personnelles ou confidentielles.

## Développement
```bash
pip install -r requirements-dev.txt
python -m pytest
```
Structure : `streamlit_app.py` (interface) · `utils_data.py` (profiling, nettoyage) · `charts.py` (graphiques Plotly) · `utils_openai.py` (IA) · `utils_pdf.py` (rapports) · `styles.py` (CSS).
