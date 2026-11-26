
import numpy as np
import pandas as pd
import pytest

import utils_data


class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def test_numeric_columns_not_detected_as_datetime():
    assert utils_data.detect_column_type(pd.Series([1, 2, 3, 4, 5])) == "numeric"
    assert utils_data.detect_column_type(pd.Series([1.5, 2.5, 3.5])) == "numeric"


def test_date_strings_detected_as_datetime():
    s = pd.Series(["2024-01-01", "2024-02-03", "2024-03-05", "2024-04-07"])
    assert utils_data.detect_column_type(s) == "datetime"


def test_empty_and_binary():
    assert utils_data.detect_column_type(pd.Series([np.nan, np.nan])) == "empty"
    assert utils_data.detect_column_type(pd.Series(["a", "b", "a", "b"])) == "binary"


def test_profile_on_empty_dataframe_does_not_crash():
    profil = utils_data.profile_data_quality(pd.DataFrame())
    assert profil["rows"] == 0
    assert profil["total_outliers"] == 0


def test_profile_basic_counts():
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"], "c": [0, 0, 0, 0]})
    profil = utils_data.profile_data_quality(df)
    assert profil["total_missing"] == 1
    assert profil["constant_columns"] == ["c"]
    assert 0 <= profil["global_score"] <= 100


def test_load_csv_semicolon_latin1():
    raw = "nom;ville\nJosé;Orléans\n".encode("latin-1")
    df = utils_data.load_dataframe(FakeUpload("x.csv", raw))
    assert list(df.columns) == ["nom", "ville"]
    assert df.iloc[0, 0] == "José"


def test_load_unsupported_raises():
    with pytest.raises(ValueError):
        utils_data.load_dataframe(FakeUpload("x.txt", b"abc"))


def test_clean_dataframe():
    df = pd.DataFrame({"a": [1, 1, 2], "c": [0, 0, 0]})
    out = utils_data.clean_dataframe(df)
    assert list(out.columns) == ["a"] and len(out) == 2


def test_pdf_and_html_reports_escape_and_build():
    import utils_pdf

    df = pd.DataFrame({"<b>x</b>": [None, None, None], "ok": [1, 2, 3]})
    profil = utils_data.profile_data_quality(df)
    html = utils_pdf.generate_pdf_html("<script>alert(1)</script>", profil, df)
    assert "<script>alert(1)</script>" not in html
    assert "<b>x</b>" not in html
    assert utils_pdf.generate_pdf_bytes("## Synthèse **ok**", profil).startswith(b"%PDF")


def _sample_profile():
    df = pd.DataFrame({
        "id": range(10),
        "vide": [None] * 10,
        "const": ["x"] * 10,
        "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
    })
    return df, utils_data.profile_data_quality(df)


def test_top_issues_sorted_by_severity():
    _, profil = _sample_profile()
    issues = utils_data.top_issues(profil, n=5)
    severities = [i["severity"] for i in issues]
    assert severities == sorted(severities, reverse=True)
    assert issues[0]["severity"] == 3
    assert len(utils_data.top_issues(profil, n=2)) == 2


def test_column_summary_status():
    df, profil = _sample_profile()
    summary = utils_data.build_column_summary(df, profil).set_index("Colonne")
    assert summary.loc["vide", "Statut"] == "Critique"
    assert summary.loc["const", "Statut"] == "À surveiller"
    assert summary.loc["id", "Statut"] == "OK"
    assert "outlier" in summary.loc["val", "Alertes"]


def test_score_label():
    assert utils_data.score_label(95) == "Bonne qualité"
    assert utils_data.score_label(10) == "Qualité critique"


def test_clean_dataframe_options():
    df = pd.DataFrame({"a": [1, 1, 2], "const": [0, 0, 0], "sparse": [None, None, 1], "t": [" x ", " x ", "y"]})
    out = utils_data.clean_dataframe(df, drop_constant=False, drop_duplicates=False)
    assert "const" in out.columns and len(out) == 3
    out = utils_data.clean_dataframe(df, max_missing_pct=50, strip_text=True)
    assert "sparse" not in out.columns and "const" not in out.columns
    assert out["t"].tolist() == ["x", "y"]  # espaces retirés puis doublons supprimés
    assert df["t"].iloc[0] == " x "  # original intact


def test_get_problem_rows():
    df = pd.DataFrame({"a": [1, 1, 2, None], "v": [1, 1, 1, 100.0]})
    profil = utils_data.profile_data_quality(df)
    assert len(utils_data.get_problem_rows(df, profil, "duplicates")) == 2
    assert len(utils_data.get_problem_rows(df, profil, "missing", "a")) == 1
    assert len(utils_data.get_problem_rows(df, profil, "missing")) == 1
    assert utils_data.get_problem_rows(df, profil, "outliers", "v").index.tolist() == [3]
    with pytest.raises(ValueError):
        utils_data.get_problem_rows(df, profil, "nope")


def test_charts_build():
    import charts

    df = pd.DataFrame({"a": [1, 2, 3, None, 100.0], "b": [1, 2, 3, 4, 5], "c": list("xyxyz")})
    profil = utils_data.profile_data_quality(df)
    assert charts.create_missing_chart(profil) is not None
    assert charts.create_types_chart(profil["column_types"]) is not None
    assert charts.create_outliers_chart(profil) is not None
    assert charts.create_correlation_chart(profil["correlations"]) is not None
    charts.create_numeric_charts(df["a"], "a")
    fig, vc = charts.create_category_chart(df["c"], "c")
    assert len(vc) == 3
