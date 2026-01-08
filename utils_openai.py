# utils_openai.py

from openai import OpenAI
from typing import Dict, Optional
import pandas as pd


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Crée un client OpenAI.
    
    Args:
        api_key: Clé API OpenAI (optionnel si définie en variable d'environnement)
        
    Returns:
        OpenAI: Client OpenAI initialisé
    """
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()  # Utilise la variable d'environnement OPENAI_API_KEY


def generate_synthesis(client: OpenAI, df: pd.DataFrame, profil: Dict) -> str:
    """
    Génère une synthèse professionnelle de la qualité des données via IA.
    
    Args:
        client: Client OpenAI
        df: DataFrame analysé
        profil: Dictionnaire de profil de qualité
        
    Returns:
        str: Synthèse en markdown
    """
    # Créer un échantillon du schéma avec exemples
    schema_samples = []
    for col in df.columns[:20]:  # Limiter à 20 colonnes pour le prompt
        sample_values = df[col].dropna().head(3).tolist()
        dtype = profil["dtypes"][col]
        missing_pct = profil["missing_pct"][col]
        
        schema_samples.append(
            f"- **{col}** ({dtype}): {sample_values} | Valeurs manquantes: {missing_pct}%"
        )
    
    schema_text = "\n".join(schema_samples)
    
    # Construire le contexte détaillé
    context = f"""
## DATASET ANALYSÉ

**Dimensions:**
- Lignes: {profil['rows']:,}
- Colonnes: {profil['cols']}
- Taille mémoire: {profil.get('memory_usage', 0):.2f} MB

**Qualité globale:**
- Score global: {profil['global_score']}%
- Taux de valeurs manquantes: {profil['missing_rate']:.2f}%
- Nombre total de valeurs manquantes: {profil['total_missing']:,}
- Lignes dupliquées: {profil['duplicate_rows']} ({profil['duplicate_rate']:.2f}%)

**Distribution des types de colonnes:**
{profil.get('type_distribution', {})}

**Problèmes identifiés:**
- Colonnes constantes: {len(profil['constant_columns'])}
- Colonnes vides: {len(profil['empty_columns'])}
- Colonnes à haute cardinalité: {len(profil.get('high_cardinality_cols', []))}
- Total d'outliers: {profil.get('total_outliers', 0)}

**Échantillon du schéma (premières colonnes):**
{schema_text}

**Scores détaillés:**
{profil.get('score_details', {})}
"""

    prompt = f"""
Tu es un **consultant expert en Data Quality et Data Engineering** avec 15 ans d'expérience.

Voici le contexte complet d'un dataset que tu dois analyser:

{context}

**TA MISSION:**

1) **Synthèse Professionnelle** (12-18 lignes)
   - Commence par une évaluation globale du score de qualité
   - Identifie les 3 problèmes majeurs par ordre de criticité
   - Explique l'impact métier de ces problèmes
   - Utilise un ton professionnel mais accessible
   - Utilise des emojis pertinents pour la lisibilité (⚠️ 🔍 ✅ 📊)

2) **Tableau de Priorisation**
   Crée un tableau markdown avec ces colonnes:
   | Priorité | Problème | Colonnes concernées | Impact | Recommandation |
   
   Inclus 5-7 lignes par ordre de priorité décroissante.

3) **Quick Wins** (5 actions rapides)
   Liste 5 actions concrètes et immédiatement applicables avec:
   - 🎯 Action
   - 💡 Bénéfice attendu
   - ⏱️ Effort estimé (Faible/Moyen/Élevé)

4) **Tests de Qualité Recommandés**
   Suggère 3-5 tests automatisés à mettre en place

**FORMAT DE RÉPONSE:**
Utilise le markdown avec des sections claires, des tableaux, et des listes à puces.
Sois concret, actionnable et professionnel.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Tu es un consultant expert en Data Quality avec une expertise approfondie en analyse de données, data engineering et gouvernance des données. Tu fournis des analyses professionnelles, actionnables et orientées business."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ **Erreur lors de la génération de la synthèse:**\n\n{str(e)}"


def generate_tests(client: OpenAI, df: pd.DataFrame) -> str:
    """
    Génère des tests de qualité de données adaptés au dataset.
    
    Args:
        client: Client OpenAI
        df: DataFrame à tester
        
    Returns:
        str: Liste de tests en markdown
    """
    # Créer un aperçu du schéma
    schema_info = []
    for col in df.columns[:15]:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        missing = df[col].isna().sum()
        sample = df[col].dropna().head(2).tolist()
        
        schema_info.append(
            f"- **{col}** ({dtype}): {nunique} valeurs uniques, {missing} NaN | Ex: {sample}"
        )
    
    schema_text = "\n".join(schema_info)
    
    prompt = f"""
Tu es un expert en **Data Quality Testing** et en automatisation de tests.

Voici le schéma d'un dataset à tester:

**Dimensions:** {len(df)} lignes × {len(df.columns)} colonnes

**Colonnes:**
{schema_text}

**TA MISSION:**

Propose **8 tests de qualité des données** spécifiquement adaptés à ce dataset.

Pour CHAQUE test, fournis:

### Test N: [Nom descriptif du test]

- **Objectif:** Pourquoi ce test est important
- **Critère de succès:** Conditions précises pour passer le test
- **Colonnes concernées:** Liste des colonnes à tester
- **Sévérité:** Critique / Majeure / Mineure
- **Code Python (exemple):**
```python
# Code de test concret et exécutable
```

**TYPES DE TESTS À COUVRIR:**
1. Complétude (valeurs manquantes)
2. Validité (format, plage de valeurs)
3. Cohérence (relations entre colonnes)
4. Unicité (doublons, clés)
5. Exactitude (valeurs aberrantes)
6. Conformité (règles métier)

Sois **TRÈS SPÉCIFIQUE** aux colonnes et types de ce dataset.
Fournis du code Python **EXÉCUTABLE** utilisant pandas.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en tests de qualité de données et en validation de datasets. Tu crées des tests concrets, exécutables et adaptés au contexte spécifique de chaque dataset."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.6
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ **Erreur lors de la génération des tests:**\n\n{str(e)}"


def generate_cleaning_script(client: OpenAI, df: pd.DataFrame, profil: Dict) -> str:
    """
    Génère un script Python complet de nettoyage des données.
    
    Args:
        client: Client OpenAI
        df: DataFrame à nettoyer
        profil: Profil de qualité
        
    Returns:
        str: Script Python commenté
    """
    problems = []
    
    if profil["empty_columns"]:
        problems.append(f"Colonnes vides: {profil['empty_columns']}")
    
    if profil["constant_columns"]:
        problems.append(f"Colonnes constantes: {profil['constant_columns']}")
    
    if profil["duplicate_rows"] > 0:
        problems.append(f"{profil['duplicate_rows']} lignes dupliquées")
    
    high_missing = profil["missing_pct"][profil["missing_pct"] > 30]
    if not high_missing.empty:
        problems.append(f"Colonnes avec >30% de NaN: {high_missing.index.tolist()}")
    
    problems_text = "\n".join([f"- {p}" for p in problems])
    
    prompt = f"""
Génère un script Python complet et exécutable pour nettoyer ce dataset.

**Problèmes identifiés:**
{problems_text}

**Dimensions:** {profil['rows']} lignes × {profil['cols']} colonnes

Le script doit:
1. Être entièrement commenté
2. Utiliser pandas
3. Gérer chaque problème identifié
4. Inclure des vérifications avant/après
5. Être exécutable tel quel

Format: Code Python pur avec commentaires détaillés.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en data cleaning et préparation de données. Tu écris du code Python propre, commenté et exécutable."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"# Erreur lors de la génération du script:\n# {str(e)}"
