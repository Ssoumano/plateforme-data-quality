# utils_data.py

import pandas as pd
import numpy as np
import io

def detect_separator(uploaded_file_bytes: bytes) -> str:
    sample = uploaded_file_bytes[:4096].decode(errors="ignore")
    for sep in [';', ',', '\t', '|']:
        if sep in sample:
            return sep
    return ','

def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        sep = detect_separator(data)
        return pd.read_csv(io.BytesIO(data), sep=sep, engine="python")
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def safe_numeric(df):
    """Sélectionne uniquement les colonnes numériques, même si aucune."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric if numeric.shape[1] > 0 else pd.DataFrame()


def profile_data_quality(df: pd.DataFrame) -> dict:
    profil = {}

    profil["rows"] = len(df)
    profil["cols"] = df.shape[1]

    profil["missing_count"] = df.isna().sum()
    profil["missing_pct"] = (df.isna().mean() * 100).round(2)
    profil["dtypes"] = df.dtypes.astype(str)

    profil["constant_columns"] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    profil["empty_columns"] = [c for c in df.columns if df[c].dropna().shape[0] == 0]
    profil["duplicate_rows"] = int(df.duplicated().sum())

    numeric = safe_numeric(df)

    if numeric.empty:
        profil["numeric_stats"] = pd.DataFrame()
        profil["outliers"] = {}
    else:
        profil["numeric_stats"] = numeric.describe().T

        outliers = {}
        for col in numeric.columns:
            series = df[col].dropna()
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers[col] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        profil["outliers"] = outliers

    miss_score = max(0, 100 - profil["missing_pct"].mean())
    dup_score = max(0, 100 - (profil["duplicate_rows"] / max(1, len(df))) * 100)
    out_score = max(0, 100 - (np.mean(list(profil["outliers"].values())) if profil["outliers"] else 0))

    profil["global_score"] = round(miss_score * 0.5 + dup_score * 0.3 + out_score * 0.2, 1)

    return profil
