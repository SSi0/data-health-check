"""The Streamlit page, driven with Streamlit's AppTest harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import COMPANY_XLSX, CUSTOMERS_CSV, PASSWORD_XLSX

APP = str(Path(__file__).parents[1] / "app.py")
PII_TOGGLE = "Detect sensitive data types (PII)"
REVEAL_TOGGLE = "Show sensitive values"
TIMEOUT = 120


def start(detect_pii: bool) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    app.toggle(key=PII_TOGGLE).set_value(detect_pii).run()
    return app


def upload(app: AppTest, path: Path, name: str | None = None) -> AppTest:
    app.file_uploader[0].upload(name or path.name, path.read_bytes())
    return app.run()


def all_text(app: AppTest) -> str:
    parts = [e.value for e in (*app.markdown, *app.caption, *app.info)]
    return "\n".join(str(part) for part in parts)


def test_page_shows_instructions_privacy_notice_and_footer():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not app.exception
    assert app.title[0].value == "Data Health Check"
    notice = app.info[0].value
    assert notice.startswith("Your file is processed on a hosted server, in memory")
    assert "Please don't upload confidential or regulated data." in notice
    text = all_text(app)
    assert "What you'll get" in [h.value for h in app.subheader]
    assert "Profiling powered by DataProfiler (Apache 2.0)." in text
    assert "Not affiliated with Capital One." in text
    assert "(https://saltbridgesystems.com)" in text


def test_csv_upload_in_lite_mode_offers_csv_download():
    app = upload(start(detect_pii=False), CUSTOMERS_CSV)
    assert not app.exception
    assert not app.error
    assert "has been profiled" in app.success[0].value
    metrics = {m.label: m.value for m in app.metric}
    assert metrics == {"Rows": "200", "Columns": "16", "Mode": "Lite"}
    assert len(app.dataframe) == 1
    assert "data_label" not in app.dataframe[0].value.columns
    button = app.get("download_button")[0]
    assert "customers_profile.csv" in button.proto.label


def test_lite_mode_hides_samples_on_screen_unless_revealed():
    app = upload(start(detect_pii=False), CUSTOMERS_CSV)
    table = app.dataframe[0].value.set_index("column_name")
    assert set(table["sample_values"]) == {"[hidden: Lite mode]"}
    assert any("hidden in Lite mode" in c.value for c in app.caption)

    app.toggle(key=REVEAL_TOGGLE).set_value(True).run()
    table = app.dataframe[0].value.set_index("column_name")
    assert table.loc["ssn", "sample_values"].count("-") == 10


def test_upload_is_cleared_after_profiling():
    app = upload(start(detect_pii=False), CUSTOMERS_CSV)
    assert app.file_uploader[0].value is None
    assert app.session_state["profile"].file_name == "customers.csv"


def test_xlsx_upload_offers_xlsx_download_and_one_table_per_sheet():
    app = upload(start(detect_pii=False), COMPANY_XLSX)
    assert not app.error
    assert [s.value for s in app.subheader][-2:] == [
        "Sheet: Customers",
        "Sheet: Orders",
    ]
    assert len(app.dataframe) == 2
    assert {m.label: m.value for m in app.metric}["Sheets"] == "2"
    button = app.get("download_button")[0]
    assert "company_data_profile.xlsx" in button.proto.label


@pytest.mark.parametrize(
    ("path", "name", "message"),
    [
        (CUSTOMERS_CSV, "customers.txt", "isn't supported"),
        (CUSTOMERS_CSV, "customers.xls", "isn't supported"),
        (PASSWORD_XLSX, None, "password-protected"),
    ],
)
def test_bad_files_show_a_friendly_error(path, name, message):
    app = upload(start(detect_pii=False), path, name)
    assert not app.exception
    assert message in app.error[0].value
    assert not app.success


def test_empty_file_shows_a_friendly_error():
    app = start(detect_pii=False)
    app.file_uploader[0].upload("empty.csv", b"")
    app.run()
    assert "empty" in app.error[0].value


@pytest.mark.full_mode
def test_full_mode_masks_sensitive_samples_unless_revealed():
    app = upload(start(detect_pii=True), CUSTOMERS_CSV)
    assert not app.error
    table = app.dataframe[0].value.set_index("column_name")
    assert table.loc["ssn", "data_label"] == "SSN"
    assert table.loc["ssn", "sample_values"] == "[masked: SSN]"
    assert table.loc["plan", "sample_values"] != "[masked: SSN]"

    app.toggle(key=REVEAL_TOGGLE).set_value(True).run()
    table = app.dataframe[0].value.set_index("column_name")
    assert table.loc["ssn", "sample_values"].count("-") == 10


def test_pii_detection_defaults_off_with_an_explanation_when_memory_is_short(
    monkeypatch,
):
    from data_health_check import profiler

    monkeypatch.setattr(profiler, "full_mode_fits", lambda: False)
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert app.toggle(key=PII_TOGGLE).value is False
    explanation = app.sidebar.info[0].value
    assert "off by default because this server has about" in explanation
    assert "run the app on your own computer" in explanation


def test_pii_detection_defaults_on_when_memory_allows(monkeypatch):
    from data_health_check import profiler

    monkeypatch.setattr(profiler, "full_mode_fits", lambda: True)
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert app.toggle(key=PII_TOGGLE).value is True
    assert not app.sidebar.info


def test_pii_toggle_is_disabled_when_tensorflow_is_missing(monkeypatch):
    from data_health_check import profiler

    monkeypatch.setattr(profiler, "pii_detection_available", lambda: False)
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    toggle = app.toggle(key=PII_TOGGLE)
    assert toggle.value is False
    assert toggle.disabled
    assert "TensorFlow isn't installed" in app.sidebar.info[0].value
