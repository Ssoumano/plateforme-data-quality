# utils_openai.py

from openai import OpenAI

def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

def generate_synthesis(client, df, profil):
    schema = "\n".join([f"- {col}: {str(df[col].head(5).tolist())[:80]}..." for col in df.columns])

    prompt = f"""
Tu es consultant expert en Data Quality.

Données :
- Lignes : {profil['rows']}
- Colonnes : {profil['cols']}
- Valeurs manquantes moyennes : {profil['missing_pct'].mean():.2f}%
- Doublons : {profil['duplicate_rows']}
- Outliers : {profil['outliers']}
- Colonnes constantes : {profil['constant_columns']}

Schéma simplifié :
{schema}

1) Rédige une synthèse professionnelle (10-15 lignes).
2) Crée un tableau Priorité | Problème | Colonnes | Recommandation.
3) Donne 5 Quick Wins.
"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Tu es un consultant expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=900
    )
    return r.choices[0].message.content


def generate_tests(client, df):
    schema = "\n".join([f"- {c}: {str(df[c].head().tolist())[:60]}..." for c in df.columns])

    prompt = f"Propose 8 tests de data quality adaptés au dataset suivant :\n{schema}"

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert en data quality."},
            {"role": "user", "content": prompt}
        ]
    )
    return r.choices[0].message.content
