from __future__ import annotations

import functools
from pathlib import Path

import pytest

from data_health_check import profiler

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS_CSV = FIXTURES / "customers.csv"
COMPANY_XLSX = FIXTURES / "company_data.xlsx"
PASSWORD_XLSX = FIXTURES / "password_protected.xlsx"


@pytest.fixture(scope="session")
def get_labeler():
    """A cached labeler loader, shared by all full-mode tests."""
    return functools.cache(profiler.load_labeler)


@pytest.fixture(scope="session")
def csv_profile_lite() -> profiler.DatasetProfile:
    return profiler.profile_file(
        CUSTOMERS_CSV.read_bytes(), CUSTOMERS_CSV.name, detect_pii=False
    )


@pytest.fixture(scope="session")
def xlsx_profile_lite() -> profiler.DatasetProfile:
    return profiler.profile_file(
        COMPANY_XLSX.read_bytes(), COMPANY_XLSX.name, detect_pii=False
    )


@pytest.fixture(scope="session")
def csv_profile_full(get_labeler) -> profiler.DatasetProfile:
    return profiler.profile_file(
        CUSTOMERS_CSV.read_bytes(), CUSTOMERS_CSV.name, get_labeler=get_labeler
    )


@pytest.fixture(scope="session")
def xlsx_profile_full(get_labeler) -> profiler.DatasetProfile:
    return profiler.profile_file(
        COMPANY_XLSX.read_bytes(), COMPANY_XLSX.name, get_labeler=get_labeler
    )


def columns_by_name(sheet: profiler.SheetProfile) -> dict[str, dict]:
    return {column["column_name"]: column for column in sheet.columns}


def digits_only_ids_csv() -> bytes:
    """Fake SSNs, phones and cards stored as plain digits (typed as numbers)."""
    lines = ["ssn,phone,card,quantity"]
    for i in range(200):
        lines.append(
            f"{900000000 + i * 4999},{2025550100 + i % 100},"
            f"{4111111111111111 + i * 2},{i % 12 + 1}"
        )
    return ("\n".join(lines) + "\n").encode()
