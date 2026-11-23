import io

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
