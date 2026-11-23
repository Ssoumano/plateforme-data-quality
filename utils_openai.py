
from openai import OpenAI
from typing import Dict, Optional
import pandas as pd

DEFAULT_MODEL = "gpt-4o-mini"


def _samples(series: pd.Series, n: int) -> list:
    """n valeurs non nulles tirées au hasard (reproductible)."""
    values = series.dropna()
    if len(values) > n:
        values = values.sample(n, random_state=0)
    return values.tolist()


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Crée un client OpenAI."""
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()  # Utilise la variable d'environnement OPENAI_API_KEY


def generate_synthesis(client: OpenAI, df: pd.DataFrame, profil: Dict,
                       include_samples: bool = False, model: str = DEFAULT_MODEL) -> str:
    """Génère une synthèse professionnelle de la qualité des données via IA."""
    schema_samples = []
    for col in df.columns[:20]:  # Limiter à 20 colonnes pour le prompt
        dtype = profil["dtypes"][col]
        missing_pct = profil["missing_pct"][col]
        examples = f" | Exemples: {_samples(df[col], 3)}" if include_samples else ""

        schema_samples.append(
            f"- **{col}** ({dtype}) | Valeurs manquantes: {missing_pct}%{examples}"
        )

    schema_text = "\n".join(schema_samples)

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
Tu es spécialiste de la qualité des données.

Voici le contexte complet d'un dataset que tu dois analyser:

{context}

**TA MISSION:**

1) **Synthèse Professionnelle** (12-18 lignes)
   - Commence par une évaluation globale du score de qualité
   - Identifie les 3 problèmes majeurs par ordre de criticité
   - Explique l'impact métier de ces problèmes
   - Utilise un ton professionnel mais accessible

2) **Tableau de Priorisation**
   Crée un tableau markdown avec ces colonnes:
   | Priorité | Problème | Colonnes concernées | Impact | Recommandation |

   Inclus 5-7 lignes par ordre de priorité décroissante.

3) **Quick Wins** (5 actions rapides)
   Liste 5 actions concrètes et immédiatement applicables avec:
   - Action
   - Bénéfice attendu
   - ⏱ Effort estimé (Faible/Moyen/Élevé)

4) **Tests de Qualité Recommandés**
   Suggère 3-5 tests automatisés à mettre en place

**FORMAT DE RÉPONSE:**
Utilise le markdown avec des sections claires, des tableaux, et des listes à puces.
Sois concret et précis.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Tu analyses la qualité de jeux de données et proposes des corrections concrètes."
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


def generate_tests(client: OpenAI, df: pd.DataFrame,
                   include_samples: bool = False, model: str = DEFAULT_MODEL) -> str:
    """Génère des tests de qualité de données adaptés au dataset."""
    schema_info = []
    for col in df.columns[:15]:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        missing = df[col].isna().sum()
        examples = f" | Ex: {_samples(df[col], 2)}" if include_samples else ""

        schema_info.append(
            f"- **{col}** ({dtype}): {nunique} valeurs uniques, {missing} NaN{examples}"
        )

    schema_text = "\n".join(schema_info)

    prompt = f"""
Tu écris des tests de qualité de données.

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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Tu écris des tests de qualité de données concrets, exécutables et adaptés au jeu de données fourni."
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


def generate_cleaning_script(client: OpenAI, df: pd.DataFrame, profil: Dict,
                             model: str = DEFAULT_MODEL) -> str:
    """Génère un script Python complet de nettoyage des données."""
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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Tu écris du code Python pandas de nettoyage de données, commenté et exécutable."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500,
        temperature=0.5
    )

    code = response.choices[0].message.content.strip()
    if code.startswith("```"):  # retire les clôtures markdown
        code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("```"))
    return code
