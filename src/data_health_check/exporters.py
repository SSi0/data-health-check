"""Write a profile in the same format as the input file (CSV or XLSX)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from data_health_check.profiler import FIELD_GUIDE, DatasetProfile

CSV_MIME_TYPE = "text/csv"
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

SUMMARY_SHEET = "Summary"
FIELD_GUIDE_SHEET = "Field guide"
SUMMARY_HEADERS = (
    "File name",
    "Sheet name",
    "Rows",
    "Columns",
    "Rows profiled",
    "Profiled at (UTC)",
    "Mode",
    "Notes",
)

_EXCEL_MAX_CELL_CHARS = 32767
_EXCEL_MAX_SHEET_NAME = 31
_EXCEL_SHEET_NAME_FORBIDDEN = str.maketrans({c: "_" for c in "[]:*?/\\"})
_MIN_WIDTH, _MAX_WIDTH = 8, 60
# Cells starting with these can be run as formulas by spreadsheet apps.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


@dataclass(frozen=True)
class ExportedProfile:
    file_name: str
    data: bytes
    mime_type: str


def output_file_name(input_name: str) -> str:
    """``customers.xlsx`` -> ``customers_profile.xlsx``."""
    path = PurePath(input_name)
    return f"{path.stem}_profile{path.suffix.lower()}"


def export_profile(
    profile: DatasetProfile, *, reveal_sensitive: bool = False
) -> ExportedProfile:
    """Render ``profile`` in the input's format, masking sensitive samples."""
    name = output_file_name(profile.file_name)
    if profile.file_format == "csv":
        return ExportedProfile(
            name,
            to_csv_bytes(profile, reveal_sensitive=reveal_sensitive),
            CSV_MIME_TYPE,
        )
    return ExportedProfile(
        name, to_xlsx_bytes(profile, reveal_sensitive=reveal_sensitive), XLSX_MIME_TYPE
    )


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def to_csv_bytes(profile: DatasetProfile, *, reveal_sensitive: bool = False) -> bytes:
    """One row per input column. UTF-8 with a BOM so Excel shows accents."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(profile.column_fields()))
    writer.writeheader()
    for sheet in profile.sheets:
        for row in profile.column_rows(sheet, reveal_sensitive=reveal_sensitive):
            writer.writerow({key: _csv_safe(value) for key, value in row.items()})
    return buffer.getvalue().encode("utf-8-sig")


def _csv_safe(value: Any) -> Any:
    """Prefix text that a spreadsheet app could run as a formula with ``'``."""
    if (
        isinstance(value, str)
        and value.startswith(_FORMULA_PREFIXES)
        and not _is_numeric_text(value)
    ):
        return "'" + value
    return value


def _is_numeric_text(value: str) -> bool:
    """True for ``-12.5`` or ``-3; 4.25`` (numbers that only look like formulas)."""
    try:
        for part in value.split(";"):
            float(part)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------


def to_xlsx_bytes(profile: DatasetProfile, *, reveal_sensitive: bool = False) -> bytes:
    """Summary sheet, one profile sheet per input sheet, and a Field guide."""
    workbook = Workbook()
    workbook.properties.creator = "Data Health Check"

    summary = workbook.active
    summary.title = SUMMARY_SHEET
    _write_table(
        summary,
        SUMMARY_HEADERS,
        [
            (
                profile.file_name,
                sheet.sheet_name,
                sheet.row_count,
                sheet.column_count,
                sheet.rows_profiled,
                profile.profiled_at,
                profile.mode,
                sheet.note,
            )
            for sheet in profile.sheets
        ],
    )

    used_names = {SUMMARY_SHEET.lower(), FIELD_GUIDE_SHEET.lower()}
    fields = profile.column_fields()
    for sheet in profile.sheets:
        title = _unique_sheet_title(sheet.sheet_name or "Profile", used_names)
        rows = profile.column_rows(sheet, reveal_sensitive=reveal_sensitive)
        _write_table(
            workbook.create_sheet(title),
            fields,
            [[row.get(name) for name in fields] for row in rows],
        )

    guide = workbook.create_sheet(FIELD_GUIDE_SHEET)
    _write_table(
        guide,
        ("Field", "What it means"),
        [(name, FIELD_GUIDE[name]) for name in fields],
        wrap_columns=(2,),
        max_width=100,
    )

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _write_table(
    sheet: Worksheet,
    headers: tuple[str, ...] | list[str],
    rows: list[Any],
    *,
    wrap_columns: tuple[int, ...] = (),
    max_width: int = _MAX_WIDTH,
) -> None:
    """Write a table with a bold, frozen header row and fitted column widths."""
    sheet.append(list(headers))
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"

    widths = [len(str(header)) for header in headers]
    for row_index, row in enumerate(rows, start=2):
        for col_index, value in enumerate(row, start=1):
            cell = sheet.cell(row=row_index, column=col_index, value=_xlsx_safe(value))
            if isinstance(value, str) and cell.data_type == "f":
                cell.data_type = "s"  # text that starts with "=" stays text
            if col_index in wrap_columns:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            if value is not None:
                widths[col_index - 1] = max(widths[col_index - 1], len(str(value)))

    for col_index, width in enumerate(widths, start=1):
        letter = get_column_letter(col_index)
        sheet.column_dimensions[letter].width = min(
            max(width + 2, _MIN_WIDTH), max_width
        )


def _xlsx_safe(value: Any) -> Any:
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)[:_EXCEL_MAX_CELL_CHARS]
    return value


def _unique_sheet_title(name: str, used_lower: set[str]) -> str:
    """Make a valid, unique Excel sheet title (31 chars, no []:*?/\\)."""
    base = ILLEGAL_CHARACTERS_RE.sub("", name).translate(_EXCEL_SHEET_NAME_FORBIDDEN)
    base = base.strip().strip("'")[:_EXCEL_MAX_SHEET_NAME] or "Profile"
    title, counter = base, 2
    while title.lower() in used_lower:
        suffix = f" ({counter})"
        title = base[: _EXCEL_MAX_SHEET_NAME - len(suffix)] + suffix
        counter += 1
    used_lower.add(title.lower())
    return title
