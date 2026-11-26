
import pandas as pd
import numpy as np
import io
import warnings
from typing import Dict, Optional, Tuple

# Seuils heuristiques, à ajuster selon le domaine.
DATE_PARSE_RATIO = 0.75         # part de valeurs parsables en date pour typer "datetime"
CATEGORICAL_RATIO = 0.05        # nb de valeurs distinctes / nb de lignes sous lequel on parle de catégorie
HIGH_CARDINALITY_RATIO = 0.95   # au-delà, la colonne ressemble à un identifiant
IQR_FACTOR = 1.5                # facteur de la règle de Tukey pour les outliers

# Poids du score global ; la somme vaut 1.
SCORE_WEIGHTS = {"missing": 0.40, "duplicates": 0.25, "outliers": 0.20, "problematic_columns": 0.15}


def detect_separator(uploaded_file_bytes: bytes) -> str:
    """Détecte automatiquement le séparateur d'un fichier CSV."""
    sample = uploaded_file_bytes[:4096].decode(errors="ignore")
    separators = [';', ',', '\t', '|']

    counts = {sep: sample.count(sep) for sep in separators}

    if max(counts.values()) > 0:
        return max(counts, key=counts.get)
    return ','


def load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Charge un DataFrame depuis un fichier uploadé (CSV, Excel).

    Args:
        uploaded_file: Fichier Streamlit uploadé

    Returns:
        pd.DataFrame, ou None si aucun fichier.

    Raises:
        ValueError / pandas errors si le fichier est illisible.
    """
    if uploaded_file is None:
        return None

    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        sep = detect_separator(data)
        # Encodage : UTF-8 (avec BOM) d'abord, puis latin-1 sans perte silencieuse
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Encodage du fichier CSV non reconnu.")
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(data))
    raise ValueError("Format non supporté (CSV, XLS ou XLSX attendu).")


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne uniquement les colonnes numériques du DataFrame."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric if numeric.shape[1] > 0 else pd.DataFrame()


def detect_column_type(series: pd.Series) -> str:
    """Détecte le type sémantique d'une colonne."""
    if series.isna().all():
        return "empty"

    # Types déjà natifs : traités avant toute conversion
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Colonnes texte : tenter une conversion en date
    # (les entiers ne sont plus convertis en timestamps par erreur)
    non_null = series.dropna()
    if len(non_null) > 0:
        try:
            sample = non_null.astype(str).head(1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conv = pd.to_datetime(sample, errors="coerce")
            if conv.notna().mean() > DATE_PARSE_RATIO:
                return "datetime"
        except (ValueError, TypeError, OverflowError):
            pass

    unique_count = series.nunique(dropna=True)
    if unique_count == 2:
        return "binary"

    if unique_count / max(1, len(series)) < CATEGORICAL_RATIO:
        return "categorical"

    return "text"


def calculate_outliers(series: pd.Series) -> Tuple[int, list]:
    """Calcule le nombre d'outliers et retourne leurs indices."""
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return 0, []

    q1 = series_clean.quantile(0.25)
    q3 = series_clean.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - IQR_FACTOR * iqr
    upper_bound = q3 + IQR_FACTOR * iqr

    outliers_mask = (series_clean < lower_bound) | (series_clean > upper_bound)
    outlier_indices = series_clean[outliers_mask].index.tolist()

    return int(outliers_mask.sum()), outlier_indices


def profile_data_quality(df: pd.DataFrame) -> Dict:
    """Profile complet de la qualité des données d'un DataFrame."""
    profil = {}

    profil["rows"] = len(df)
    profil["cols"] = df.shape[1]
    profil["memory_usage"] = df.memory_usage(deep=True).sum() / 1024**2  # En MB

    profil["missing_count"] = df.isna().sum()
    profil["missing_pct"] = (df.isna().mean() * 100).round(2)
    profil["total_missing"] = int(df.isna().sum().sum())
    profil["missing_rate"] = round((profil["total_missing"] / max(1, df.shape[0] * df.shape[1])) * 100, 2)

    profil["dtypes"] = df.dtypes.astype(str)
    profil["column_types"] = {col: detect_column_type(df[col]) for col in df.columns}

    type_counts = pd.Series(profil["column_types"]).value_counts().to_dict()
    profil["type_distribution"] = type_counts

    profil["constant_columns"] = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 1
    ]

    profil["empty_columns"] = [
        c for c in df.columns
        if df[c].dropna().shape[0] == 0
    ]

    profil["duplicate_rows"] = int(df.duplicated().sum())
    profil["duplicate_rate"] = round((profil["duplicate_rows"] / max(1, len(df))) * 100, 2)

    profil["high_cardinality_cols"] = [
        col for col in df.columns
        if df[col].nunique() / max(1, len(df)) > HIGH_CARDINALITY_RATIO and profil["column_types"][col] not in ['numeric', 'datetime']
    ]

    numeric = safe_numeric(df)

    if numeric.empty:
        profil["numeric_stats"] = pd.DataFrame()
        profil["outliers"] = {}
        profil["outlier_indices"] = {}
        profil["total_outliers"] = 0
    else:
        profil["numeric_stats"] = numeric.describe().T

        outliers = {}
        outlier_indices = {}
        for col in numeric.columns:
            count, indices = calculate_outliers(df[col])
            outliers[col] = count
            outlier_indices[col] = indices

        profil["outliers"] = outliers
        profil["outlier_indices"] = outlier_indices
        profil["total_outliers"] = sum(outliers.values())

    if not numeric.empty and numeric.shape[1] > 1:
        profil["correlations"] = numeric.corr()
    else:
        profil["correlations"] = pd.DataFrame()


    miss_score = max(0, 100 - profil["missing_rate"])

    dup_score = max(0, 100 - profil["duplicate_rate"])

    if profil["outliers"]:
        # rapporté au nombre de cellules numériques, pas au nombre de lignes
        n_numeric_cells = max(1, len(df) * len(profil["outliers"]))
        outlier_rate = (profil["total_outliers"] / n_numeric_cells) * 100
        out_score = max(0, 100 - outlier_rate)
    else:
        out_score = 100

    prob_cols = len(profil["constant_columns"]) + len(profil["empty_columns"])
    prob_score = max(0, 100 - (prob_cols / max(1, profil["cols"])) * 100)

    profil["global_score"] = round(
        miss_score * SCORE_WEIGHTS["missing"]
        + dup_score * SCORE_WEIGHTS["duplicates"]
        + out_score * SCORE_WEIGHTS["outliers"]
        + prob_score * SCORE_WEIGHTS["problematic_columns"],
        1
    )

    profil["score_details"] = {
        "missing_score": round(miss_score, 1),
        "duplicate_score": round(dup_score, 1),
        "outlier_score": round(out_score, 1),
        "problematic_columns_score": round(prob_score, 1)
    }

    recommendations = []

    if profil["missing_rate"] > 10:
        recommendations.append("Taux élevé de valeurs manquantes détecté (>10%)")

    if profil["duplicate_rate"] > 5:
        recommendations.append("Taux élevé de doublons détecté (>5%)")

    if len(profil["constant_columns"]) > 0:
        recommendations.append(f"ℹ {len(profil['constant_columns'])} colonne(s) constante(s) à supprimer")

    if len(profil["empty_columns"]) > 0:
        recommendations.append(f"ℹ {len(profil['empty_columns'])} colonne(s) vide(s) à supprimer")

    if len(profil["high_cardinality_cols"]) > 0:
        recommendations.append(f"ℹ {len(profil['high_cardinality_cols'])} colonne(s) à haute cardinalité détectée(s)")

    profil["recommendations"] = recommendations if recommendations else ["Aucun problème majeur détecté"]

    return profil


def clean_dataframe(df: pd.DataFrame, drop_empty: bool = True, drop_constant: bool = True,
                    drop_duplicates: bool = True, max_missing_pct: Optional[float] = None,
                    strip_text: bool = False) -> pd.DataFrame:
    """Nettoyage configurable. Ne modifie pas le DataFrame d'origine.

    Args:
        drop_empty: supprime les colonnes entièrement vides
        drop_constant: supprime les colonnes à valeur unique (vides comprises)
        drop_duplicates: supprime les lignes dupliquées
        max_missing_pct: supprime les colonnes dépassant ce % de valeurs manquantes
        strip_text: retire les espaces en début/fin des colonnes texte
    """
    out = df.copy()
    to_drop = set()
    for col in out.columns:
        nunique = out[col].nunique(dropna=True)
        if drop_empty and nunique == 0:
            to_drop.add(col)
        if drop_constant and nunique <= 1:
            to_drop.add(col)
        if max_missing_pct is not None and out[col].isna().mean() * 100 > max_missing_pct:
            to_drop.add(col)
    out = out.drop(columns=list(to_drop))

    if strip_text:
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    if drop_duplicates:
        out = out.drop_duplicates()
    return out


def get_problem_rows(df: pd.DataFrame, profil: Dict, kind: str,
                     column: Optional[str] = None) -> pd.DataFrame:
    """Lignes concernées par un problème de qualité.

    Args:
        kind: "duplicates", "missing" ou "outliers"
        column: colonne ciblée (obligatoire pour "outliers", optionnelle pour "missing")
    """
    if kind == "duplicates":
        rows = df[df.duplicated(keep=False)]
        try:
            return rows.sort_values(list(df.columns))
        except TypeError:
            return rows
    if kind == "missing":
        mask = df[column].isna() if column else df.isna().any(axis=1)
        return df[mask]
    if kind == "outliers":
        if column is None:
            raise ValueError("column requis pour kind='outliers'")
        return df.loc[profil.get("outlier_indices", {}).get(column, [])]
    raise ValueError(f"kind inconnu : {kind}")


def score_label(score: float) -> str:
    """Libellé qualitatif d'un score global (0-100)."""
    if score >= 80:
        return "Bonne qualité"
    if score >= 60:
        return "Qualité moyenne"
    if score >= 40:
        return "Qualité dégradée"
    return "Qualité critique"


def top_issues(profil: Dict, n: int = 3) -> list:
    """Problèmes les plus importants, du plus grave au moins grave.

    Returns:
        list de dicts {"severity": 1-3 (3 = critique), "title", "detail"}
    """
    issues = []
    missing_pct = profil["missing_pct"]

    if profil["empty_columns"]:
        issues.append({"severity": 3, "title": f"{len(profil['empty_columns'])} colonne(s) entièrement vide(s)",
                       "detail": ", ".join(map(str, profil["empty_columns"][:5]))})
    high_missing = [c for c in missing_pct.index if missing_pct[c] > 50 and c not in profil["empty_columns"]]
    if high_missing:
        issues.append({"severity": 3, "title": f"{len(high_missing)} colonne(s) avec plus de 50 % de valeurs manquantes",
                       "detail": ", ".join(map(str, high_missing[:5]))})
    if profil["duplicate_rate"] > 5:
        issues.append({"severity": 3, "title": f"{profil['duplicate_rows']:,} lignes dupliquées",
                       "detail": f"{profil['duplicate_rate']:.1f} % du fichier"})
    elif profil["duplicate_rows"] > 0:
        issues.append({"severity": 2, "title": f"{profil['duplicate_rows']:,} lignes dupliquées",
                       "detail": f"{profil['duplicate_rate']:.1f} % du fichier"})
    if profil["missing_rate"] > 10:
        issues.append({"severity": 2, "title": f"{profil['missing_rate']:.1f} % de cellules manquantes",
                       "detail": f"{profil['total_missing']:,} valeurs"})
    if profil["constant_columns"]:
        issues.append({"severity": 2, "title": f"{len(profil['constant_columns'])} colonne(s) constante(s)",
                       "detail": ", ".join(map(str, profil["constant_columns"][:5]))})
    cols_out = sorted(profil.get("outliers", {}).items(), key=lambda kv: -kv[1])
    cols_out = [(c, n_) for c, n_ in cols_out if n_ > 0]
    if cols_out:
        issues.append({"severity": 1, "title": f"{profil.get('total_outliers', 0):,} valeurs aberrantes (IQR)",
                       "detail": ", ".join(f"{c} ({n_})" for c, n_ in cols_out[:3])})
    if profil["high_cardinality_cols"]:
        issues.append({"severity": 1, "title": f"{len(profil['high_cardinality_cols'])} colonne(s) à haute cardinalité",
                       "detail": ", ".join(map(str, profil["high_cardinality_cols"][:5]))})

    issues.sort(key=lambda i: -i["severity"])
    return issues[:n]


def build_column_summary(df: pd.DataFrame, profil: Dict) -> pd.DataFrame:
    """Tableau récapitulatif : une ligne par colonne, avec statut et alertes."""
    rows = []
    for col in df.columns:
        missing = float(profil["missing_pct"][col])
        ctype = profil["column_types"][col]
        n_out = int(profil.get("outliers", {}).get(col, 0))
        alerts = []
        severity = 0
        if col in profil["empty_columns"]:
            alerts.append("vide"); severity = 3
        elif col in profil["constant_columns"]:
            alerts.append("constante"); severity = max(severity, 2)
        if missing > 50 and col not in profil["empty_columns"]:
            alerts.append(">50 % manquant"); severity = 3
        elif missing > 10:
            alerts.append(">10 % manquant"); severity = max(severity, 2)
        if n_out > 0:
            alerts.append(f"{n_out} outlier(s)"); severity = max(severity, 1)
        if col in profil["high_cardinality_cols"]:
            alerts.append("haute cardinalité"); severity = max(severity, 1)
        status = {3: "Critique", 2: "À surveiller", 1: "Mineur", 0: "OK"}[severity]
        rows.append({
            "Colonne": str(col), "Statut": status, "Type": ctype,
            "% manquant": missing, "Uniques": int(df[col].nunique(dropna=True)),
            "Outliers": n_out, "Alertes": ", ".join(alerts) or "—",
        })
    return pd.DataFrame(rows)


def generate_quick_wins(profil: Dict) -> list:
    """Génère une liste de Quick Wins basée sur le profil."""
    quick_wins = []

    if profil["empty_columns"]:
        quick_wins.append({
            "priority": "Haute",
            "action": "Supprimer les colonnes vides",
            "columns": profil["empty_columns"],
            "impact": "Réduction immédiate de la taille du dataset",
            "code": f"df = df.drop(columns={profil['empty_columns']})"
        })

    if profil["constant_columns"]:
        quick_wins.append({
            "priority": "Haute",
            "action": "Supprimer les colonnes constantes",
            "columns": profil["constant_columns"],
            "impact": "Amélioration de la pertinence des données",
            "code": f"df = df.drop(columns={profil['constant_columns']})"
        })

    if profil["duplicate_rows"] > 0:
        quick_wins.append({
            "priority": "Haute",
            "action": f"Supprimer {profil['duplicate_rows']} ligne(s) en double",
            "columns": "Toutes",
            "impact": f"Réduction de {profil['duplicate_rate']}% du dataset",
            "code": "df = df.drop_duplicates()"
        })

    high_missing = profil["missing_pct"][profil["missing_pct"] > 50]
    if not high_missing.empty:
        quick_wins.append({
            "priority": "Moyenne",
            "action": "Traiter les colonnes avec >50% de valeurs manquantes",
            "columns": high_missing.index.tolist(),
            "impact": "Amélioration de la complétude des données",
            "code": "# Options: imputation, suppression ou collecte de nouvelles données"
        })

    if "text" in profil.get("type_distribution", {}):
        quick_wins.append({
            "priority": "Moyenne",
            "action": "Standardiser les colonnes textuelles",
            "columns": [k for k, v in profil["column_types"].items() if v == "text"],
            "impact": "Amélioration de la cohérence des données",
            "code": "df['col'] = df['col'].str.strip().str.lower()"
        })

    return quick_wins[:5]  # Limiter à 5 quick wins
