"""Profiling engine shared by the web app and the MCP server.

Files are handled as bytes in memory and are never written to disk.
:func:`profile_file` reads a CSV or XLSX file, runs DataProfiler on each table
(the CSV, or each worksheet) and flattens DataProfiler's report into one plain
dict per input column, using the field names in :data:`COLUMN_FIELDS`.
"""

from __future__ import annotations

import codecs
import contextlib
import csv
import importlib.util
import io
import logging
import math
import os
import sys
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

import pandas as pd

from data_health_check import masking

# Silence TensorFlow's C++ start-up messages; only effective before TF loads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Library warnings can quote cell values (openpyxl: "serial value 968226991 is
# outside the limits for dates"). warnings.catch_warnings() is not thread-safe,
# so with two files profiled at once one run could restore the other's filters
# mid-read; these filters are therefore kept for the whole process.
_QUIET_MODULES = (r"openpyxl(\..*)?", r"dataprofiler(\..*)?", r"pandas(\..*)?")


def quiet_library_warnings() -> None:
    """Ignore warnings from the file-reading libraries, process-wide.

    Idempotent. Called at import and before every run, because a host may
    reset the warning filters (pytest does after collecting tests). Once in
    place, a later ``catch_warnings()`` block saves and restores them too.
    """
    # A filter's module is a compiled pattern, a plain string or None.
    present = {(f[0], getattr(f[3], "pattern", f[3])) for f in warnings.filters}
    for module in _QUIET_MODULES:
        if ("ignore", module) not in present:
            warnings.filterwarnings("ignore", module=module)


quiet_library_warnings()

MAX_FILE_MB = 25
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = (".csv", ".xlsx")

MODE_FULL = "full"
MODE_LITE = "lite"

MAX_SAMPLE_VALUES = 5
MAX_SAMPLE_CHARS = 100
SAMPLE_SEPARATOR = "; "

# Output fields, in output order. The two label fields exist only in full mode.
COLUMN_FIELDS: tuple[str, ...] = (
    "column_name",
    "data_type",
    "data_label",
    "data_label_confidence",
    "null_count",
    "null_percent",
    "unique_count",
    "unique_ratio",
    "min",
    "max",
    "mean",
    "median",
    "stddev",
    "sample_values",
    "rows_profiled",
)
LABEL_FIELDS: tuple[str, ...] = ("data_label", "data_label_confidence")

FIELD_GUIDE: dict[str, str] = {
    "column_name": "The column's name, taken from the first (header) row of your file.",
    "data_type": (
        "The kind of values in the column: int (whole numbers), float (decimal "
        "numbers), datetime (dates or times) or string (text)."
    ),
    "data_label": (
        "What the values appear to be, detected by a machine-learning model: for "
        "example EMAIL_ADDRESS, PHONE_NUMBER, SSN, ADDRESS, CREDIT_CARD, INTEGER, "
        "or UNKNOWN when it can't tell. Two labels joined by | mean both were "
        "about equally likely. Full mode only."
    ),
    "data_label_confidence": (
        "How sure the model is about the data label (the first one, if two are "
        "shown), from 0 (not sure) to 1 (certain). Full mode only."
    ),
    "null_count": (
        "How many rows have no value in this column: blank, or written as null, "
        "none or nan, or only spaces, dashes or underscores."
    ),
    "null_percent": "The empty rows as a percentage of the rows profiled (0 to 100).",
    "unique_count": "How many different (distinct) non-empty values the column has.",
    "unique_ratio": (
        "unique_count divided by the number of non-empty values. 1 means every "
        "value is different (typical for IDs); close to 0 means a few values "
        "repeat a lot (typical for categories)."
    ),
    "min": (
        "The smallest value. Number and date columns only. Like sample_values, "
        "it is masked in sensitive columns and hidden for number columns in Lite "
        "mode, because it is a real value from your data."
    ),
    "max": (
        "The largest value. Number and date columns only. Masked and hidden like min."
    ),
    "mean": (
        "The average. Number columns only. Masked in sensitive columns, and "
        "hidden in Lite mode when a column has two different values or fewer, "
        "because then it would give the values away."
    ),
    "median": (
        "The middle value: half the values are below it. Number columns only; "
        "DataProfiler estimates it, so it can be slightly approximate. Masked and "
        "hidden like min."
    ),
    "stddev": (
        "Standard deviation: how spread out the numbers are around the average. "
        "Number columns only. Masked and hidden like mean."
    ),
    "sample_values": (
        "Up to 5 example values, separated by '; '. In columns detected as "
        "sensitive (such as emails, phone numbers or SSNs) they are replaced by "
        "[masked: LABEL]. Lite mode can't detect sensitive columns, so there all "
        "examples are replaced by [hidden: Lite mode], unless whoever made the "
        "profile chose to show them. Detection can miss columns (names are often "
        "labelled UNKNOWN), so treat unmasked examples as possibly personal."
    ),
    "rows_profiled": (
        "How many rows were analysed. For large files DataProfiler analyses a "
        "random sample of rows, so counts and percentages describe that sample."
    ),
}

ProgressCallback = Callable[[float, str], None]

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls and encrypted Office
_ZIP_MAGIC = b"PK\x03\x04"  # .xlsx is a zip archive
_ENCRYPTED_PACKAGE = "EncryptedPackage".encode("utf-16-le")
_SNIFF_BYTES = 64 * 1024
_DECODE_CHUNK = 1 << 20

# The labeler's TensorFlow model is shared between callers, so runs using it
# are serialized. This also keeps peak memory down on small servers.
_LABELER_LOCK = threading.Lock()


class ProfilingError(Exception):
    """A problem that can be explained to the person who supplied the file.

    Messages are written in plain language and never contain file contents,
    so they are safe to display and to log.
    """


class UnsupportedFileTypeError(ProfilingError):
    pass


class FileTooLargeError(ProfilingError):
    pass


class EmptyFileError(ProfilingError):
    pass


class UnreadableFileError(ProfilingError):
    pass


class EncodingError(UnreadableFileError):
    pass


class PasswordProtectedError(UnreadableFileError):
    pass


class OutOfMemoryError(ProfilingError):
    pass


class DetectionUnavailableError(ProfilingError):
    pass


DETECTION_UNAVAILABLE_MESSAGE = (
    "Sensitive data detection isn't available on this computer because "
    "TensorFlow isn't installed (it isn't made for every type of computer, "
    "for example older Intel Macs). Use Lite mode instead."
)


OUT_OF_MEMORY_MESSAGE = (
    "There wasn't enough memory to profile this file. Try Lite mode (turn off "
    "sensitive data type detection), which needs much less memory, or try a "
    "smaller file."
)
OUT_OF_MEMORY_LITE_MESSAGE = (
    "There wasn't enough memory to profile this file, even in Lite mode. Try a "
    "smaller file (fewer rows or columns), or run Data Health Check on your own "
    "computer (see the README)."
)


@dataclass
class SheetProfile:
    """The profile of one table: the CSV file, or one worksheet."""

    sheet_name: str | None
    row_count: int
    column_count: int
    rows_profiled: int
    columns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sampled(self) -> bool:
        return self.rows_profiled < self.row_count

    @property
    def note(self) -> str:
        if self.row_count == 0:
            return "No data rows; nothing to profile."
        if self.sampled:
            return (
                f"Large table: statistics are based on a random sample of "
                f"{self.rows_profiled:,} of {self.row_count:,} rows."
            )
        return ""


@dataclass
class DatasetProfile:
    """The profile of a whole file."""

    file_name: str
    file_format: str
    mode: str
    profiled_at: str
    sheets: list[SheetProfile]

    @property
    def detect_pii(self) -> bool:
        return self.mode == MODE_FULL

    @property
    def row_count(self) -> int:
        return sum(sheet.row_count for sheet in self.sheets)

    @property
    def column_count(self) -> int:
        return sum(sheet.column_count for sheet in self.sheets)

    def column_fields(self) -> tuple[str, ...]:
        """The per-column fields present in this profile, in output order."""
        if self.detect_pii:
            return COLUMN_FIELDS
        return tuple(name for name in COLUMN_FIELDS if name not in LABEL_FIELDS)

    def column_rows(
        self, sheet: SheetProfile, *, reveal_sensitive: bool = False
    ) -> list[dict[str, Any]]:
        """Return output-ready rows (one per column) for ``sheet``.

        ``sample_values`` is joined into one string. Unless ``reveal_sensitive``
        is True, values copied from the data (samples, and min/max/median) are
        masked in sensitive columns and hidden in Lite mode, following
        :mod:`data_health_check.masking`.
        """
        rows = []
        for column in sheet.columns:
            if not reveal_sensitive:
                column = masking.protect_column(column, labels_checked=self.detect_pii)
            row = {name: column.get(name) for name in self.column_fields()}
            row["sample_values"] = SAMPLE_SEPARATOR.join(
                column.get("sample_values") or []
            )
            rows.append(row)
        return rows

    def to_dict(
        self, *, include_samples: bool = False, reveal_sensitive: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-serializable summary of the profile.

        Fields that don't apply to a column (None) are left out.
        """
        sheets = []
        for sheet in self.sheets:
            columns = []
            for row in self.column_rows(sheet, reveal_sensitive=reveal_sensitive):
                if not include_samples:
                    row.pop("sample_values", None)
                columns.append({k: v for k, v in row.items() if v is not None})
            sheets.append(
                {
                    "sheet_name": sheet.sheet_name,
                    "row_count": sheet.row_count,
                    "column_count": sheet.column_count,
                    "rows_profiled": sheet.rows_profiled,
                    "sampled": sheet.sampled,
                    "note": sheet.note or None,
                    "columns": columns,
                }
            )
        return {
            "file_name": self.file_name,
            "file_format": self.file_format,
            "mode": self.mode,
            "profiled_at": self.profiled_at,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "sheet_count": len(self.sheets),
            "sheets": sheets,
        }


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------


def detect_format(file_name: str) -> str:
    """Return ``"csv"`` or ``"xlsx"`` from the file name, or raise."""
    suffix = PurePath(file_name).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return suffix[1:]
    shown = suffix or "no extension"
    raise UnsupportedFileTypeError(
        f"This file type ({shown}) isn't supported. Please use a .csv or .xlsx "
        "file. (In Excel, use File > Save As and choose 'CSV UTF-8' or "
        "'Excel Workbook (.xlsx)'.)"
    )


def check_size(size_bytes: int) -> None:
    """Raise :class:`FileTooLargeError` if the file is over the size limit."""
    if size_bytes > MAX_FILE_BYTES:
        raise FileTooLargeError(
            f"This file is {size_bytes / 1024 / 1024:.1f} MB, which is over the "
            f"{MAX_FILE_MB} MB limit. Try removing rows or columns you don't "
            "need, or run Data Health Check on your own computer (see the README)."
        )


def read_tables(data: bytes, file_format: str) -> list[tuple[str | None, pd.DataFrame]]:
    """Read ``data`` into ``(sheet_name, frame)`` pairs.

    CSV files give one pair with ``sheet_name=None``. Values are read as text
    (or raw cell values for XLSX) so that DataProfiler does its own typing
    and null detection.
    """
    try:
        with warnings.catch_warnings():
            # Reader warnings can quote cell values (e.g. openpyxl's "serial
            # value ... is outside the limits for dates"): never print them.
            warnings.simplefilter("ignore")
            if file_format == "csv":
                return [(None, _read_csv(data))]
            if file_format == "xlsx":
                return _read_xlsx(data)
    except MemoryError:
        raise OutOfMemoryError(OUT_OF_MEMORY_MESSAGE) from None
    raise UnsupportedFileTypeError(f"Unsupported format: {file_format!r}.")


_EMPTY_MESSAGE = "This file is empty. Please choose a file with data in it."


def _read_csv(data: bytes) -> pd.DataFrame:
    if not data.strip():
        raise EmptyFileError(_EMPTY_MESSAGE)
    if data.startswith((_ZIP_MAGIC, _OLE_MAGIC)):
        raise UnreadableFileError(
            "This looks like an Excel file with a .csv name. Rename it to end in "
            ".xlsx, or open it in Excel and save it as 'CSV UTF-8'."
        )
    encoding = _detect_text_encoding(data)
    delimiter = _sniff_delimiter(data[:_SNIFF_BYTES].decode(encoding, errors="ignore"))
    try:
        frame = pd.read_csv(
            io.BytesIO(data),
            sep=delimiter,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        raise EmptyFileError(_EMPTY_MESSAGE) from None
    except pd.errors.ParserError as exc:
        # pandas' message names the line and field counts, never the content.
        detail = str(exc).strip().split("\n")[0]
        raise UnreadableFileError(
            "We couldn't read this CSV file because its rows don't line up with "
            f"the header ({detail}). Check for stray commas or quotes, or re-save "
            "it from Excel as 'CSV UTF-8'."
        ) from None
    except UnicodeDecodeError:
        raise EncodingError(_ENCODING_MESSAGE) from None
    if frame.empty:
        raise EmptyFileError(
            "This file has column headers but no data rows, so there is nothing "
            "to profile."
        )
    return frame


_ENCODING_MESSAGE = (
    "We couldn't read the text in this file (its character encoding isn't "
    "recognised). Please open it in Excel or another spreadsheet app, save it "
    "as 'CSV UTF-8', and try again."
)


def _detect_text_encoding(data: bytes) -> str:
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if b"\x00" in data[:4096]:
        raise UnreadableFileError(
            "This file doesn't look like a CSV text file. Please export your data "
            "as CSV (or upload the original .xlsx file)."
        )
    # cp1252 is what Excel on Windows uses for plain "CSV" exports.
    for encoding in ("utf-8", "cp1252"):
        if _decodes_cleanly(data, encoding):
            return encoding
    raise EncodingError(_ENCODING_MESSAGE)


def _decodes_cleanly(data: bytes, encoding: str) -> bool:
    """Check ``data`` decodes, in chunks so the whole text is never held twice."""
    decoder = codecs.getincrementaldecoder(encoding)()
    view = memoryview(data)
    try:
        for start in range(0, len(view), _DECODE_CHUNK):
            decoder.decode(view[start : start + _DECODE_CHUNK])
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _sniff_delimiter(sample: str) -> str:
    # Drop the last line: it may have been cut off mid-row.
    lines = sample.splitlines()[:-1] or sample.splitlines()
    try:
        return csv.Sniffer().sniff("\n".join(lines[:100]), delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_xlsx(data: bytes) -> list[tuple[str | None, pd.DataFrame]]:
    if not data.strip():
        raise EmptyFileError(_EMPTY_MESSAGE)
    if data.startswith(_OLE_MAGIC):
        if _ENCRYPTED_PACKAGE in data:
            raise PasswordProtectedError(
                "This Excel file is password-protected, so it can't be read. In "
                "Excel, open it, go to File > Info > Protect Workbook > Encrypt "
                "with Password, delete the password, save it, and try again."
            )
        raise UnsupportedFileTypeError(
            "This looks like an older Excel file (.xls) renamed to .xlsx. Open it "
            "in Excel and use File > Save As > 'Excel Workbook (.xlsx)'."
        )
    if not data.startswith(_ZIP_MAGIC):
        raise UnreadableFileError(
            "This file isn't a valid .xlsx workbook. If it is a CSV file, rename "
            "it to end in .csv; otherwise re-save it from Excel as .xlsx."
        )
    try:
        sheets = pd.read_excel(
            io.BytesIO(data), sheet_name=None, dtype=object, engine="openpyxl"
        )
    except MemoryError:
        raise
    except Exception:
        raise UnreadableFileError(
            "We couldn't open this Excel file. It may be damaged or saved in an "
            "unusual format. Try opening it in Excel and saving it again as "
            "'Excel Workbook (.xlsx)'."
        ) from None

    tables = []
    for name, frame in sheets.items():
        tables.append((str(name), _drop_blank_edges(frame)))
    if all(frame.empty for _, frame in tables):
        raise EmptyFileError(
            "None of the sheets in this workbook contain data, so there is "
            "nothing to profile."
        )
    return tables


def _drop_blank_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely empty and empty columns with no header.

    Excel often reports formatted-but-empty rows and columns as part of the
    sheet; people don't see them as data, so they are not profiled.
    """
    frame = frame.dropna(how="all")
    keep = [
        column
        for column in frame.columns
        if not (str(column).startswith("Unnamed:") and frame[column].isna().all())
    ]
    return frame[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------


def _import_dataprofiler() -> Any:
    """Import DataProfiler lazily (it is slow to import) and quiet its logs."""
    import dataprofiler as dp

    dp.set_verbosity(logging.WARNING)
    return dp


def load_labeler() -> Any:
    """Load DataProfiler's structured data labeler (a TensorFlow model).

    This is slow and uses a lot of memory: callers should cache the result and
    pass the cached loader to :func:`profile_file` as ``get_labeler``.
    """
    dp = _import_dataprofiler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dp.DataLabeler(labeler_type="structured")


# Memory model, measured on Linux x86-64 with Python 3.12, DataProfiler 0.14.0
# and tensorflow-cpu 2.21 (peak RSS of one run, see the README):
# * importing TensorFlow and loading the labeler adds about 480 MB;
# * profiling costs about 6 MB per MB of CSV and 18 MB per MB of XLSX
#   (XLSX is compressed, so it holds far more cells per MB).
LABELER_MEMORY_MB = 500
MEMORY_PER_INPUT_MB = {"csv": 6, "xlsx": 20}
MEMORY_HEADROOM_MB = 100


def estimate_memory_mb(size_bytes: int, file_format: str, *, detect_pii: bool) -> float:
    """Estimate the extra memory (MB) this process needs to profile a file."""
    needed = size_bytes / 2**20 * MEMORY_PER_INPUT_MB[file_format] + MEMORY_HEADROOM_MB
    if detect_pii and "tensorflow" not in sys.modules:
        needed += LABELER_MEMORY_MB
    return needed


def available_memory_mb() -> float:
    """Memory (MB) this process can still use, honouring container limits."""
    import psutil

    available = psutil.virtual_memory().available
    cgroup = _cgroup_memory()
    if cgroup is not None:
        limit, working_set = cgroup
        available = min(available, max(limit - working_set, 0))
    return available / 2**20


def pii_detection_available() -> bool:
    """True if TensorFlow, which the data labeler needs, is installed."""
    return importlib.util.find_spec("tensorflow") is not None


def full_mode_fits() -> bool:
    """True if full mode can profile a maximum-size CSV with the memory left."""
    if not pii_detection_available():
        return False
    needed = estimate_memory_mb(MAX_FILE_BYTES, "csv", detect_pii=True)
    return available_memory_mb() >= needed


def check_memory(size_bytes: int, file_format: str, *, detect_pii: bool) -> None:
    """Raise :class:`OutOfMemoryError` before starting a run that can't fit.

    On small hosts a run that runs out of memory gets the whole app killed,
    so it is better to refuse it with a helpful message.
    """
    if estimate_memory_mb(size_bytes, file_format, detect_pii=detect_pii) > (
        available_memory_mb()
    ):
        raise OutOfMemoryError(OUT_OF_MEMORY_MESSAGE)


_CGROUP_FILES = (
    # (directory, limit file, usage file, reclaimable-cache key in memory.stat)
    ("/sys/fs/cgroup", "memory.max", "memory.current", "inactive_file"),
    (
        "/sys/fs/cgroup/memory",
        "memory.limit_in_bytes",
        "memory.usage_in_bytes",
        "total_inactive_file",
    ),
)


def _cgroup_memory() -> tuple[int, int] | None:
    """Return ``(limit, working_set)`` in bytes if a cgroup limit applies."""
    for directory, limit_file, usage_file, inactive_key in _CGROUP_FILES:
        base = Path(directory)
        try:
            limit_text = (base / limit_file).read_text().strip()
            usage = int((base / usage_file).read_text())
            stat = (base / "memory.stat").read_text()
        except (OSError, ValueError):
            continue
        if not limit_text.isdigit() or int(limit_text) >= 1 << 60:
            return None  # "max" or a huge number: no limit
        inactive = 0
        for line in stat.splitlines():
            key, _, value = line.partition(" ")
            if key == inactive_key and value.isdigit():
                inactive = int(value)
        return int(limit_text), usage - inactive
    return None


def profile_file(
    data: bytes,
    file_name: str,
    *,
    detect_pii: bool = True,
    get_labeler: Callable[[], Any] | None = None,
    progress: ProgressCallback | None = None,
) -> DatasetProfile:
    """Profile a CSV or XLSX file given as bytes.

    :param data: the file's contents.
    :param file_name: the original file name; its extension picks the format.
    :param detect_pii: full mode (run the data labeler) if True, lite if False.
    :param get_labeler: returns the labeler in full mode. Pass a cached loader
        (e.g. ``functools.cache(load_labeler)``) so the model loads only once;
        defaults to loading a fresh one.
    :param progress: optional ``callback(fraction, message)`` for progress.
    :raises ProfilingError: with a user-friendly message for bad input.
    """
    quiet_library_warnings()
    step = progress or (lambda fraction, message: None)
    file_format = detect_format(file_name)
    check_size(len(data))
    if detect_pii and not pii_detection_available():
        raise DetectionUnavailableError(DETECTION_UNAVAILABLE_MESSAGE)
    try:
        check_memory(len(data), file_format, detect_pii=detect_pii)

        step(0.05, "Reading your file…")
        tables = read_tables(data, file_format)

        labeler = None
        if detect_pii:
            step(0.10, "Loading the sensitive data detection model…")
            labeler = (get_labeler or load_labeler)()

        sheets = []
        for index, (sheet_name, frame) in enumerate(tables):
            where = f"sheet '{sheet_name}'" if sheet_name is not None else "your data"
            step(0.15 + 0.8 * index / len(tables), f"Profiling {where}…")
            sheets.append(profile_table(frame, sheet_name, labeler))
    except (OutOfMemoryError, MemoryError):
        message = OUT_OF_MEMORY_MESSAGE if detect_pii else OUT_OF_MEMORY_LITE_MESSAGE
        raise OutOfMemoryError(message) from None
    step(1.0, "Done.")

    return DatasetProfile(
        file_name=PurePath(file_name).name,
        file_format=file_format,
        mode=MODE_FULL if detect_pii else MODE_LITE,
        profiled_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sheets=sheets,
    )


def profile_table(
    frame: pd.DataFrame, sheet_name: str | None = None, labeler: Any = None
) -> SheetProfile:
    """Profile one table. Pass a ``labeler`` for full mode, None for lite."""
    if frame.empty:
        return SheetProfile(sheet_name, len(frame), len(frame.columns), 0)

    dp = _import_dataprofiler()
    options = dp.ProfilerOptions()
    # Worker processes would duplicate memory (and the TF model) on small hosts.
    options.set({"structured_options.multiprocess.is_enabled": False})
    if labeler is None:
        options.set({"structured_options.data_labeler.is_enabled": False})
    else:
        options.set({"structured_options.data_labeler.data_labeler_object": labeler})

    lock = _LABELER_LOCK if labeler is not None else contextlib.nullcontext()
    try:
        with lock, warnings.catch_warnings():
            # DataProfiler warns about sampling and edge cases; the sampling is
            # reported in our output instead.
            warnings.simplefilter("ignore")
            profiler = dp.Profiler(frame, profiler_type="structured", options=options)
            report = profiler.report(report_options={"output_format": "serializable"})
    except MemoryError:
        raise OutOfMemoryError(OUT_OF_MEMORY_MESSAGE) from None
    except Exception as exc:
        # TensorFlow's own out-of-memory error; matched by name to avoid
        # importing TensorFlow here.
        if type(exc).__name__ == "ResourceExhaustedError":
            raise OutOfMemoryError(OUT_OF_MEMORY_MESSAGE) from None
        raise

    labeled = labeler is not None
    global_stats = report.get("global_stats") or {}
    return SheetProfile(
        sheet_name=sheet_name,
        row_count=int(global_stats.get("row_count", len(frame))),
        column_count=int(global_stats.get("column_count", len(frame.columns))),
        rows_profiled=int(global_stats.get("samples_used", len(frame))),
        columns=[_flatten_column(col, labeled) for col in report.get("data_stats", [])],
    )


def _flatten_column(column: dict[str, Any], labeled: bool) -> dict[str, Any]:
    """Map one entry of DataProfiler's ``data_stats`` to our output fields."""
    stats = column.get("statistics") or {}
    data_type = column.get("data_type")
    rows_profiled = int(stats.get("sample_size") or 0)
    null_count = _as_int(stats.get("null_count"))

    flat: dict[str, Any] = {
        "column_name": str(column.get("column_name")),
        "data_type": data_type,
    }
    if labeled:
        label = column.get("data_label")
        flat["data_label"] = label
        flat["data_label_confidence"] = _label_confidence(
            label, stats.get("avg_predictions")
        )
    flat["null_count"] = null_count
    flat["null_percent"] = (
        round(100 * null_count / rows_profiled, 2)
        if rows_profiled and null_count is not None
        else None
    )
    flat["unique_count"] = _as_int(stats.get("unique_count"))
    flat["unique_ratio"] = _as_number(stats.get("unique_ratio"))

    # For text columns DataProfiler's min/max/mean describe text *length*,
    # which isn't what these fields mean, so they are left empty.
    for key in ("min", "max", "mean", "median", "stddev"):
        flat[key] = None
    if data_type in ("int", "float"):
        for key in ("min", "max", "mean", "median", "stddev"):
            flat[key] = _as_number(stats.get(key))
        if data_type == "int":
            flat["min"], flat["max"] = _as_int(flat["min"]), _as_int(flat["max"])
    elif data_type == "datetime":
        flat["min"], flat["max"] = stats.get("min"), stats.get("max")

    samples = column.get("samples") or []
    flat["sample_values"] = [_truncate(str(v)) for v in samples[:MAX_SAMPLE_VALUES]]
    flat["rows_profiled"] = rows_profiled
    return flat


def _label_confidence(label: str | None, avg_predictions: dict | None) -> float | None:
    if not label or not avg_predictions:
        return None
    return _as_number(avg_predictions.get(label.split("|")[0]))


def _as_number(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    return None if number is None else int(round(number))


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_SAMPLE_CHARS:
        return text
    return text[: MAX_SAMPLE_CHARS - 1] + "…"
