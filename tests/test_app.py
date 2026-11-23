from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_starts_without_file():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
