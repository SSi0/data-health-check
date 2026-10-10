"""Local MCP server: lets AI agents (such as Claude) use the profiling engine.

Runs over stdio::

    python -P -m data_health_check.mcp_server

(``-P`` keeps the launch folder off ``sys.path``, so a file such as ``csv.py``
in the folder the client starts the server in can't shadow a real module.)

It reuses the same engine, masking rules and exporters as the web app. Values
copied from the data (samples, min, max, median) are masked in columns the
labeler identifies as sensitive and hidden in Lite mode (``detect_pii=false``).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from data_health_check import __version__, exporters, profiler

logger = logging.getLogger(__name__)

T = TypeVar("T")

INSTRUCTIONS = """\
Data Health Check profiles local CSV (.csv) and Excel (.xlsx) files: for every
column it reports the data type, how many values are empty or unique, basic
statistics and, with detect_pii=true, what the values appear to be (for example
EMAIL_ADDRESS, PHONE_NUMBER or SSN) with a confidence score.

Use profile_dataset to answer questions about a file's structure, data quality
or personal data. Use export_profile when the user wants the profile saved as a
file. Pass absolute file paths ("~" is expanded).

Privacy: sample values are off by default. In columns the labeler identifies
as sensitive, sample values and min/max/median are replaced by
"[masked: LABEL]"; in Lite mode (detect_pii=false) every sample, and
min/max/median of number columns, is "[hidden: Lite mode]". Detection is
automatic and can miss things: names are often labelled UNKNOWN, free text is
not checked, and column names are taken from the file's first row as they are.
Treat any unmasked value as possibly personal: request samples only when you
need them, and don't repeat values to the user unless they ask.
"""

server = MCPServer(
    name="data-health-check",
    title="Data Health Check",
    version=__version__,
    instructions=INSTRUCTIONS,
    log_level="WARNING",
)

GENERIC_ERROR = (
    "Something went wrong while profiling this file. Check that it opens "
    "correctly in a spreadsheet app, or try again with detect_pii=false."
)
PERMISSION_ERROR = (
    "Permission denied: the server isn't allowed to read or write there. On a "
    "Mac, the app that runs this server may need access to that folder (System "
    "Settings > Privacy & Security > Files and Folders)."
)

_labeler_lock = threading.Lock()


@functools.cache
def _load_labeler() -> Any:
    return profiler.load_labeler()


def get_labeler() -> Any:
    """The data labeler, loaded once per server process on first use."""
    with _labeler_lock:
        return _load_labeler()


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

FilePath = Annotated[
    str,
    Field(
        description='Absolute path to a local .csv or .xlsx file, up to 25 MB; "~" '
        "is expanded. Every sheet of a workbook is profiled."
    ),
]
DetectPii = Annotated[
    bool,
    Field(
        description="True (Full mode): a machine-learning labeler adds data_label "
        "and data_label_confidence to every column with values, e.g. EMAIL_ADDRESS "
        "or SSN; the first call loads the model and can take up to a minute. "
        "False (Lite mode): faster and lighter, but sensitive data can't be "
        "detected, so every sample and min/max/median of number columns are hidden."
    ),
]
IncludeSamples = Annotated[
    bool,
    Field(
        description="Include up to 5 example values per column. In columns labelled "
        'as sensitive they are "[masked: <LABEL>]", and in Lite mode they are all '
        '"[hidden: Lite mode]". Detection can miss columns (e.g. names labelled '
        "UNKNOWN), so treat unmasked samples as possibly personal."
    ),
]
OutputPath = Annotated[
    str | None,
    Field(
        description='Where to save the profile. Default: "<input name>_profile.'
        '<csv|xlsx>" next to the input file. May be a folder, or a file path with '
        "the same extension as the input; relative paths are resolved against "
        "the input file's folder. An existing file is replaced only if it is a "
        "profile made by this tool; the input file and any other existing file "
        "are never overwritten."
    ),
]


PROFILE_DATASET_DESCRIPTION = """\
Profile a local .csv or .xlsx file and return the results as JSON.

Use this to understand a dataset before working with it: which columns it
has, their data types, how complete they are, basic statistics, and which
columns appear to contain personal or sensitive data (emails, phone numbers,
SSNs, credit cards, addresses, IP addresses and so on). The file is read
locally and is not modified.

Sample values are off by default. In columns labelled as sensitive, samples
and min/max/median are replaced by "[masked: <LABEL>]"; in Lite mode
(detect_pii=false) every sample, and min/max/median of number columns, is
"[hidden: Lite mode]". Detection is automatic and can miss columns (names
are often labelled UNKNOWN, free text isn't checked), so treat any unmasked
value as possibly personal.

Returns:
    A summary (file_name, file_format, mode, profiled_at, row_count,
    column_count, sheet_count) and a list of sheets (sheet_name is null for
    CSV), each with row_count, column_count, rows_profiled, sampled, note
    and columns. Each column has column_name, null_count, null_percent,
    unique_count, unique_ratio and rows_profiled, and, when they apply,
    data_type, data_label, data_label_confidence, min, max, mean, median,
    stddev and sample_values (a column with no values has no data_type or
    label). Large tables are profiled on a random sample of rows; then
    sampled is true and counts describe the sample.
"""

EXPORT_PROFILE_DESCRIPTION = """\
Profile a local .csv or .xlsx file and save the profile as a file.

Use this when the user wants a profile they can open or share. The profile
has the same format as the input: a .csv input gives a .csv profile with one
row per column; an .xlsx input gives an .xlsx profile with a Summary sheet,
one profile sheet per input sheet and a Field guide sheet. In the file,
samples and min/max/median of sensitive columns are "[masked: <LABEL>]", and
in Lite mode (detect_pii=false) all samples, and min/max/median of number
columns, are "[hidden: Lite mode]".

It never overwrites the input file or an existing file that isn't a profile
made by this tool; an earlier profile at the same path is replaced.

Returns:
    The absolute path of the saved profile file.
"""


@server.tool(
    description=PROFILE_DATASET_DESCRIPTION,
    annotations=ToolAnnotations(
        title="Profile a CSV or Excel file",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def profile_dataset(
    file_path: FilePath,
    detect_pii: DetectPii = True,
    include_samples: IncludeSamples = False,
) -> dict[str, Any]:
    """Profile a file and return JSON (see PROFILE_DATASET_DESCRIPTION)."""

    def run() -> dict[str, Any]:
        given, real = _input_path(file_path)
        profile = _profile(given, real, detect_pii)
        return profile.to_dict(include_samples=include_samples)

    return _run(run)


@server.tool(
    description=EXPORT_PROFILE_DESCRIPTION,
    annotations=ToolAnnotations(
        title="Save a profile file next to a CSV or Excel file",
        read_only_hint=False,
        destructive_hint=True,  # it can replace an earlier profile
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def export_profile(
    file_path: FilePath, output_path: OutputPath = None, detect_pii: DetectPii = True
) -> str:
    """Profile a file and save it (see EXPORT_PROFILE_DESCRIPTION)."""

    def run() -> str:
        given, real = _input_path(file_path)
        target = _output_path(given, real, output_path)
        exported = exporters.export_profile(_profile(given, real, detect_pii))
        _write_atomically(target, exported.data)
        return str(target)

    return _run(run)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _run(action: Callable[[], T]) -> T:
    """Run ``action``, turning every failure into a ToolError the agent can read.

    Unexpected exceptions are reported generically and logged by type only:
    their messages can quote values from the file.
    """
    try:
        return action()
    except ToolError:
        raise
    except profiler.ProfilingError as exc:
        raise ToolError(str(exc)) from None
    except PermissionError:
        raise ToolError(PERMISSION_ERROR) from None
    except OSError as exc:
        reason = exc.strerror or "input/output error"
        raise ToolError(f"Couldn't access the file: {reason}.") from None
    except Exception as exc:
        logger.error("Tool failed with %s", type(exc).__name__)
        raise ToolError(GENERIC_ERROR) from None


def _profile(given: Path, real: Path, detect_pii: bool) -> profiler.DatasetProfile:
    size = real.stat().st_size
    if size > profiler.MAX_FILE_BYTES:  # check before reading it into memory
        raise ToolError(
            f"This file is {size / 1024 / 1024:.1f} MB, over the "
            f"{profiler.MAX_FILE_MB} MB limit. Ask the user to remove rows or "
            "columns they don't need, or to split it into smaller files."
        )
    return profiler.profile_file(
        real.read_bytes(), given.name, detect_pii=detect_pii, get_labeler=get_labeler
    )


def _absolute(path_text: str, base: Path | None = None) -> Path:
    """Expand "~" and make ``path_text`` absolute without following links."""
    try:
        path = Path(path_text).expanduser()
    except RuntimeError:  # "~someone" with no such user, or no home folder
        raise ToolError(
            f"Couldn't find the home folder in '{path_text}'. Use a full path."
        ) from None
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return Path(os.path.normpath(path))


def _input_path(file_path: str) -> tuple[Path, Path]:
    """Return the input as given (for its name and folder) and resolved (to read).

    The given path decides the format and where profiles go, so a link named
    ``customers.csv`` behaves like that name even if it points elsewhere.
    """
    given = _absolute(file_path)
    if given.suffix.lower() not in profiler.SUPPORTED_EXTENSIONS:
        raise ToolError(
            f"Only .csv and .xlsx files are supported, and '{given.name}' is "
            "neither. Ask the user to save it as CSV or Excel Workbook (.xlsx)."
        )
    real = given.resolve()
    if not real.exists():
        raise ToolError(f"File not found: {given}")
    if not real.is_file():
        raise ToolError(f"{given} is not a regular file.")
    return given, real


def _output_path(given: Path, real: Path, output_path: str | None) -> Path:
    suffix = given.suffix.lower()
    default_name = exporters.output_file_name(given.name)
    if output_path is None:
        target = given.parent / default_name
    else:
        target = _absolute(output_path, base=given.parent)
        if target.is_dir():
            target = target / default_name
        elif target.suffix.lower() != suffix:
            raise ToolError(
                f"output_path must end in {suffix}, the same format as the input "
                "file (or be an existing folder)."
            )
    # Follow links, so the checks below apply to the file that would be written.
    target = target.resolve()
    if target == real or (target.exists() and os.path.samefile(target, real)):
        raise ToolError("output_path can't be the input file itself.")
    if target.is_dir():
        raise ToolError(f"{target} is a folder. Choose a file name for the profile.")
    if not target.parent.is_dir():
        raise ToolError(f"The folder {target.parent} doesn't exist.")
    if target.exists() and not exporters.is_profile_file(target):
        raise ToolError(
            f"{target} already exists and isn't a profile made by Data Health "
            "Check, so it won't be overwritten. Choose another output_path."
        )
    return target


def _write_atomically(target: Path, data: bytes) -> None:
    """Write ``data`` to a temporary file next to ``target``, then swap it in.

    A failed write (for example a full disk) leaves any earlier file intact
    instead of a truncated one.
    """
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
