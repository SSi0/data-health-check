"""Profiling and report flattening."""

import io
import json
import subprocess
import sys

import pytest

from conftest import CUSTOMERS_CSV, columns_by_name
from data_health_check import profiler
from fixtures.generate_fixtures import customer_rows

LITE_FIELDS = tuple(f for f in profiler.COLUMN_FIELDS if f not in profiler.LABEL_FIELDS)


# --- Lite mode (no TensorFlow) -----------------------------------------------


def test_lite_mode_does_not_load_tensorflow():
    # In a fresh interpreter: other tests in this session may load TensorFlow.
    code = (
        "import sys; from data_health_check import profiler; "
        f"p = profiler.profile_file(open({str(CUSTOMERS_CSV)!r}, 'rb').read(), "
        "'c.csv', detect_pii=False); "
        "print(p.mode, 'tensorflow' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["lite", "False"]


def test_csv_summary(csv_profile_lite):
    profile = csv_profile_lite
    assert profile.file_name == "customers.csv"
    assert profile.file_format == "csv"
    assert (profile.row_count, profile.column_count) == (200, 16)
    sheet = profile.sheets[0]
    assert sheet.sheet_name is None
    assert sheet.rows_profiled == 200
    assert not sheet.sampled
    assert sheet.note == ""


def test_lite_mode_omits_label_fields(csv_profile_lite):
    assert csv_profile_lite.column_fields() == LITE_FIELDS
    for column in csv_profile_lite.sheets[0].columns:
        assert "data_label" not in column
        assert "data_label_confidence" not in column


def test_numeric_column(csv_profile_lite):
    column = columns_by_name(csv_profile_lite.sheets[0])["customer_id"]
    assert column["data_type"] == "int"
    assert (column["min"], column["max"]) == (10000, 10199)
    assert isinstance(column["min"], int)
    assert column["mean"] == pytest.approx(10099.5)
    assert column["unique_count"] == 200
    assert column["unique_ratio"] == 1.0
    assert column["stddev"] > 0


def test_nulls_are_counted(csv_profile_lite):
    column = columns_by_name(csv_profile_lite.sheets[0])["account_balance"]
    assert column["data_type"] == "float"
    assert column["null_count"] == 16
    assert column["null_percent"] == 8.0
    assert column["min"] < 0 < column["max"]


def test_text_columns_have_no_numeric_statistics(csv_profile_lite):
    column = columns_by_name(csv_profile_lite.sheets[0])["plan"]
    assert column["data_type"] == "string"
    for key in ("min", "max", "mean", "median", "stddev"):
        assert column[key] is None
    assert column["unique_count"] == 4


def test_date_columns_have_min_and_max(csv_profile_lite):
    column = columns_by_name(csv_profile_lite.sheets[0])["signup_date"]
    assert column["data_type"] == "datetime"
    assert column["min"] <= column["max"]
    assert column["min"].startswith("2025-")
    assert column["mean"] is None


def test_sample_values_are_capped(csv_profile_lite):
    for column in csv_profile_lite.sheets[0].columns:
        assert 1 <= len(column["sample_values"]) <= profiler.MAX_SAMPLE_VALUES


def test_xlsx_profiles_every_sheet(xlsx_profile_lite):
    profile = xlsx_profile_lite
    assert profile.file_format == "xlsx"
    names = [sheet.sheet_name for sheet in profile.sheets]
    assert names == ["Customers", "Orders"]
    customers, orders = profile.sheets
    assert (customers.row_count, customers.column_count) == (150, 16)
    assert (orders.row_count, orders.column_count) == (250, 9)
    assert columns_by_name(customers)["account_balance"]["null_count"] == 15
    assert columns_by_name(orders)["quantity"]["data_type"] == "int"


def test_large_tables_are_sampled_and_say_so():
    rows = customer_rows(6000)
    buffer = io.StringIO()
    buffer.write(",".join(rows[0]) + "\n")
    for row in rows:
        buffer.write(",".join(str(v) for v in row.values()) + "\n")
    profile = profiler.profile_file(
        buffer.getvalue().encode(), "big.csv", detect_pii=False
    )
    sheet = profile.sheets[0]
    assert sheet.row_count == 6000
    assert sheet.rows_profiled < 6000
    assert sheet.sampled
    assert "random sample" in sheet.note
    column = columns_by_name(sheet)["account_balance"]
    assert column["rows_profiled"] == sheet.rows_profiled
    expected = round(100 * column["null_count"] / sheet.rows_profiled, 2)
    assert column["null_percent"] == expected


def test_progress_is_reported():
    calls = []
    profiler.profile_file(
        b"a,b\n1,2\n3,4\n",
        "tiny.csv",
        detect_pii=False,
        progress=lambda fraction, message: calls.append((fraction, message)),
    )
    fractions = [fraction for fraction, _ in calls]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_to_dict_is_json_and_hides_samples_by_default(xlsx_profile_lite):
    result = json.loads(json.dumps(xlsx_profile_lite.to_dict()))
    assert result["mode"] == "lite"
    assert result["sheet_count"] == 2
    assert result["row_count"] == 400
    for sheet in result["sheets"]:
        for column in sheet["columns"]:
            assert "sample_values" not in column
            assert None not in column.values()


def test_to_dict_can_include_samples(csv_profile_lite):
    result = csv_profile_lite.to_dict(include_samples=True)
    column = result["sheets"][0]["columns"][0]
    assert column["column_name"] == "customer_id"
    assert len(column["sample_values"].split("; ")) == 5


# --- Memory safeguards -------------------------------------------------------


def test_estimate_grows_with_file_size_and_format():
    small = profiler.estimate_memory_mb(1 << 20, "csv", detect_pii=False)
    large = profiler.estimate_memory_mb(20 << 20, "csv", detect_pii=False)
    xlsx = profiler.estimate_memory_mb(20 << 20, "xlsx", detect_pii=False)
    assert small < large < xlsx


def test_run_that_cannot_fit_is_refused(monkeypatch):
    monkeypatch.setattr(profiler, "available_memory_mb", lambda: 50.0)
    with pytest.raises(profiler.OutOfMemoryError, match="smaller file"):
        profiler.profile_file(b"a,b\n1,2\n", "tiny.csv", detect_pii=False)


def test_full_mode_fits_follows_available_memory(monkeypatch):
    monkeypatch.setattr(profiler, "available_memory_mb", lambda: 100_000.0)
    assert profiler.full_mode_fits()
    monkeypatch.setattr(profiler, "available_memory_mb", lambda: 100.0)
    assert not profiler.full_mode_fits()


def test_full_mode_estimate_includes_tensorflow_until_it_is_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    cold = profiler.estimate_memory_mb(1 << 20, "csv", detect_pii=True)
    lite = profiler.estimate_memory_mb(1 << 20, "csv", detect_pii=False)
    assert cold - lite == profiler.LABELER_MEMORY_MB
    monkeypatch.setitem(sys.modules, "tensorflow", object())
    assert profiler.estimate_memory_mb(1 << 20, "csv", detect_pii=True) == lite


def test_available_memory_is_positive():
    assert profiler.available_memory_mb() > 0


def test_memory_errors_become_friendly(monkeypatch):
    def explode(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(profiler, "_read_csv", explode)
    with pytest.raises(profiler.OutOfMemoryError, match="even in Lite mode"):
        profiler.profile_file(b"a\n1\n", "x.csv", detect_pii=False)
    with pytest.raises(profiler.OutOfMemoryError, match="Try Lite mode"):
        profiler.profile_file(b"a\n1\n", "x.csv", get_labeler=lambda: None)


# --- Full mode (TensorFlow data labeler) ---------------------------------------


@pytest.mark.full_mode
def test_full_mode_labels_sensitive_columns(csv_profile_full):
    assert csv_profile_full.mode == profiler.MODE_FULL
    assert csv_profile_full.column_fields() == profiler.COLUMN_FIELDS
    columns = columns_by_name(csv_profile_full.sheets[0])
    expected = {
        "email": "EMAIL_ADDRESS",
        "phone": "PHONE_NUMBER",
        "ssn": "SSN",
        "credit_card": "CREDIT_CARD",
        "ip_address": "IPV4",
        "customer_id": "INTEGER",
        "signup_date": "DATE",
    }
    for name, label in expected.items():
        assert columns[name]["data_label"] == label, name
        assert 0.5 < columns[name]["data_label_confidence"] <= 1.0, name


@pytest.mark.full_mode
def test_full_mode_masks_sensitive_samples_in_rows(csv_profile_full):
    rows = {
        row["column_name"]: row
        for row in csv_profile_full.column_rows(csv_profile_full.sheets[0])
    }
    assert rows["ssn"]["sample_values"] == "[masked: SSN]"
    assert rows["email"]["sample_values"] == "[masked: EMAIL_ADDRESS]"
    assert "[masked" not in rows["plan"]["sample_values"]

    revealed = csv_profile_full.column_rows(
        csv_profile_full.sheets[0], reveal_sensitive=True
    )
    ssn = next(row for row in revealed if row["column_name"] == "ssn")
    assert ssn["sample_values"].count("-") == 10  # five raw 9xx-xx-xxxx values


@pytest.mark.full_mode
def test_full_mode_xlsx(xlsx_profile_full):
    customers, orders = xlsx_profile_full.sheets
    assert columns_by_name(customers)["ssn"]["data_label"] == "SSN"
    assert columns_by_name(orders)["contact_email"]["data_label"] == "EMAIL_ADDRESS"
