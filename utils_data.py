# utils_data.py

import pandas as pd
import numpy as np
import io
from typing import Dict, Optional, Tuple


def detect_separator(uploaded_file_bytes: bytes) -> str:
    """
    Détecte automatiquement le séparateur d'un fichier CSV.
    
    Args:
        uploaded_file_bytes: Bytes du fichier uploadé
        
    Returns:
        str: Le séparateur détecté (par défaut ',')
    """
    sample = uploaded_file_bytes[:4096].decode(errors="ignore")
    separators = [';', ',', '\t', '|']
    
    # Compte le nombre d'occurrences de chaque séparateur
    counts = {sep: sample.count(sep) for sep in separators}
    
    # Retourne le séparateur le plus fréquent, ou ',' par défaut
    if max(counts.values()) > 0:
        return max(counts, key=counts.get)
    return ','


def load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Charge un DataFrame depuis un fichier uploadé (CSV, Excel).
    
    Args:
        uploaded_file: Fichier Streamlit uploadé
        
    Returns:
        pd.DataFrame ou None si échec
    """
    if uploaded_file is None:
        return None
        
    try:
        data = uploaded_file.getvalue()
        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            sep = detect_separator(data)
            return pd.read_csv(
                io.BytesIO(data), 
                sep=sep, 
                engine="python",
                encoding_errors='ignore'
            )
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(data))
        else:
            # Tentative de lecture comme CSV par défaut
            return pd.read_csv(io.BytesIO(data), encoding_errors='ignore')
            
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        return None


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne uniquement les colonnes numériques du DataFrame.
    
    Args:
        df: DataFrame source
        
    Returns:
        pd.DataFrame contenant uniquement les colonnes numériques
    """
    numeric = df.select_dtypes(include=[np.number])
    return numeric if numeric.shape[1] > 0 else pd.DataFrame()


def detect_column_type(series: pd.Series) -> str:
    """
    Détecte le type sémantique d'une colonne.
    
    Args:
        series: Série pandas à analyser
        
    Returns:
        str: Type détecté ('empty', 'datetime', 'numeric', 'binary', 'categorical', 'text')
    """
    # Vérifier si la colonne est vide
    if series.isna().all():
        return "empty"
    
    # Vérifier si c'est une date
    try:
        conv = pd.to_datetime(series, errors="coerce")
        if conv.notna().sum() > len(series) * 0.75:
            return "datetime"
    except:
        pass
    
    # Vérifier si c'est numérique
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    # Vérifier si c'est binaire (2 valeurs uniques)
    unique_count = series.nunique(dropna=True)
    if unique_count == 2:
        return "binary"
    
    # Vérifier si c'est catégoriel (moins de 5% de valeurs uniques)
    if unique_count / max(1, len(series)) < 0.05:
        return "categorical"
    
    return "text"


def calculate_outliers(series: pd.Series) -> Tuple[int, list]:
    """
    Calcule le nombre d'outliers et retourne leurs indices.
    
    Args:
        series: Série numérique à analyser
        
    Returns:
        Tuple (nombre d'outliers, liste des indices)
    """
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return 0, []
    
    q1 = series_clean.quantile(0.25)
    q3 = series_clean.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_mask = (series_clean < lower_bound) | (series_clean > upper_bound)
    outlier_indices = series_clean[outliers_mask].index.tolist()
    
    return int(outliers_mask.sum()), outlier_indices


def profile_data_quality(df: pd.DataFrame) -> Dict:
    """
    Profile complet de la qualité des données d'un DataFrame.
    
    Args:
        df: DataFrame à profiler
        
    Returns:
        dict: Dictionnaire contenant toutes les métriques de qualité
    """
    profil = {}
    
    # ===== INFORMATIONS GÉNÉRALES =====
    profil["rows"] = len(df)
    profil["cols"] = df.shape[1]
    profil["memory_usage"] = df.memory_usage(deep=True).sum() / 1024**2  # En MB
    
    # ===== VALEURS MANQUANTES =====
    profil["missing_count"] = df.isna().sum()
    profil["missing_pct"] = (df.isna().mean() * 100).round(2)
    profil["total_missing"] = int(df.isna().sum().sum())
    profil["missing_rate"] = round((profil["total_missing"] / (df.shape[0] * df.shape[1])) * 100, 2)
    
    # ===== TYPES DE DONNÉES =====
    profil["dtypes"] = df.dtypes.astype(str)
    profil["column_types"] = {col: detect_column_type(df[col]) for col in df.columns}
    
    # Comptage par type
    type_counts = pd.Series(profil["column_types"]).value_counts().to_dict()
    profil["type_distribution"] = type_counts
    
    # ===== PROBLÈMES DE QUALITÉ =====
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
    
    # ===== COLONNES À HAUTE CARDINALITÉ =====
    profil["high_cardinality_cols"] = [
        col for col in df.columns
        if df[col].nunique() / max(1, len(df)) > 0.95 and profil["column_types"][col] not in ['numeric', 'datetime']
    ]
    
    # ===== STATISTIQUES NUMÉRIQUES =====
    numeric = safe_numeric(df)
    
    if numeric.empty:
        profil["numeric_stats"] = pd.DataFrame()
        profil["outliers"] = {}
        profil["outlier_indices"] = {}
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
    
    # ===== CORRÉLATIONS (pour colonnes numériques) =====
    if not numeric.empty and numeric.shape[1] > 1:
        profil["correlations"] = numeric.corr()
    else:
        profil["correlations"] = pd.DataFrame()
    
    # ===== SCORE GLOBAL DE QUALITÉ =====
    # Calcul pondéré basé sur plusieurs critères
    
    # Score des valeurs manquantes (0-100, plus c'est élevé, mieux c'est)
    miss_score = max(0, 100 - profil["missing_rate"])
    
    # Score des doublons
    dup_score = max(0, 100 - profil["duplicate_rate"])
    
    # Score des outliers (normalisé)
    if profil["outliers"]:
        outlier_rate = (profil["total_outliers"] / max(1, len(df))) * 100
        out_score = max(0, 100 - outlier_rate)
    else:
        out_score = 100
    
    # Score des colonnes problématiques
    prob_cols = len(profil["constant_columns"]) + len(profil["empty_columns"])
    prob_score = max(0, 100 - (prob_cols / max(1, profil["cols"])) * 100)
    
    # Score global pondéré
    profil["global_score"] = round(
        miss_score * 0.4 +      # 40% importance sur les valeurs manquantes
        dup_score * 0.25 +      # 25% importance sur les doublons
        out_score * 0.20 +      # 20% importance sur les outliers
        prob_score * 0.15,      # 15% importance sur les colonnes problématiques
        1
    )
    
    # Détails des scores par composante
    profil["score_details"] = {
        "missing_score": round(miss_score, 1),
        "duplicate_score": round(dup_score, 1),
        "outlier_score": round(out_score, 1),
        "problematic_columns_score": round(prob_score, 1)
    }
    
    # ===== RECOMMANDATIONS AUTOMATIQUES =====
    recommendations = []
    
    if profil["missing_rate"] > 10:
        recommendations.append("⚠️ Taux élevé de valeurs manquantes détecté (>10%)")
    
    if profil["duplicate_rate"] > 5:
        recommendations.append("⚠️ Taux élevé de doublons détecté (>5%)")
    
    if len(profil["constant_columns"]) > 0:
        recommendations.append(f"ℹ️ {len(profil['constant_columns'])} colonne(s) constante(s) à supprimer")
    
    if len(profil["empty_columns"]) > 0:
        recommendations.append(f"ℹ️ {len(profil['empty_columns'])} colonne(s) vide(s) à supprimer")
    
    if len(profil["high_cardinality_cols"]) > 0:
        recommendations.append(f"ℹ️ {len(profil['high_cardinality_cols'])} colonne(s) à haute cardinalité détectée(s)")
    
    profil["recommendations"] = recommendations if recommendations else ["✅ Aucun problème majeur détecté"]
    
    return profil


def generate_quick_wins(profil: Dict) -> list:
    """
    Génère une liste de Quick Wins basée sur le profil.
    
    Args:
        profil: Dictionnaire du profil de qualité
        
    Returns:
        list: Liste de quick wins avec actions concrètes
    """
    quick_wins = []
    
    # Quick Win 1: Colonnes vides
    if profil["empty_columns"]:
        quick_wins.append({
            "priority": "Haute",
            "action": "Supprimer les colonnes vides",
            "columns": profil["empty_columns"],
            "impact": "Réduction immédiate de la taille du dataset",
            "code": f"df = df.drop(columns={profil['empty_columns']})"
        })
    
    # Quick Win 2: Colonnes constantes
    if profil["constant_columns"]:
        quick_wins.append({
            "priority": "Haute",
            "action": "Supprimer les colonnes constantes",
            "columns": profil["constant_columns"],
            "impact": "Amélioration de la pertinence des données",
            "code": f"df = df.drop(columns={profil['constant_columns']})"
        })
    
    # Quick Win 3: Doublons
    if profil["duplicate_rows"] > 0:
        quick_wins.append({
            "priority": "Haute",
            "action": f"Supprimer {profil['duplicate_rows']} ligne(s) en double",
            "columns": "Toutes",
            "impact": f"Réduction de {profil['duplicate_rate']}% du dataset",
            "code": "df = df.drop_duplicates()"
        })
    
    # Quick Win 4: Valeurs manquantes critiques
    high_missing = profil["missing_pct"][profil["missing_pct"] > 50]
    if not high_missing.empty:
        quick_wins.append({
            "priority": "Moyenne",
            "action": "Traiter les colonnes avec >50% de valeurs manquantes",
            "columns": high_missing.index.tolist(),
            "impact": "Amélioration de la complétude des données",
            "code": f"# Options: imputation, suppression ou collecte de nouvelles données"
        })
    
    # Quick Win 5: Normalisation des types
    if "text" in profil.get("type_distribution", {}):
        quick_wins.append({
            "priority": "Moyenne",
            "action": "Standardiser les colonnes textuelles",
            "columns": [k for k, v in profil["column_types"].items() if v == "text"],
            "impact": "Amélioration de la cohérence des données",
            "code": "df['col'] = df['col'].str.strip().str.lower()"
        })
    
    return quick_wins[:5]  # Limiter à 5 quick wins
