"""Local MCP server: lets AI agents (such as Claude) use the profiling engine.

Runs over stdio::

    python -m data_health_check.mcp_server

It reuses the same engine, masking rules and exporters as the web app. Raw
sample values of sensitive columns are never returned to the agent, and in
Lite mode (``detect_pii=false``) no sample values are returned at all.
"""

from __future__ import annotations

import functools
import logging
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
file. Pass absolute file paths (~ is expanded). Sample values are off by
default; when requested, values of sensitive columns are returned as
"[masked: LABEL]", and in Lite mode (detect_pii=false) every sample is
"[hidden: Lite mode]". Raw sensitive values are never returned.
"""

server = MCPServer(
    name="data-health-check",
    title="Data Health Check",
    version=__version__,
    instructions=INSTRUCTIONS,
    website_url="https://saltbridgesystems.com",
    log_level="WARNING",
)

GENERIC_ERROR = (
    "Something went wrong while profiling this file. Check that it opens "
    "correctly in a spreadsheet app, or try again with detect_pii=false."
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
        "and data_label_confidence to every column, e.g. EMAIL_ADDRESS or SSN; "
        "the first call loads the model and can take up to a minute. False (Lite "
        "mode): faster and lighter, but sensitive data can't be detected."
    ),
]
IncludeSamples = Annotated[
    bool,
    Field(
        description="Include up to 5 example values per column. Samples of "
        'sensitive columns are always "[masked: <LABEL>]", and in Lite mode all '
        'samples are "[hidden: Lite mode]".'
    ),
]
OutputPath = Annotated[
    str | None,
    Field(
        description='Where to save the profile. Default: "<input name>_profile.'
        '<csv|xlsx>" next to the input file. May be a folder, or a file path with '
        "the same extension as the input; relative paths are resolved against "
        "the input file's folder. An existing file is only replaced if it is a "
        'previous profile (its name ends in "_profile.csv" or "_profile.xlsx").'
    ),
]


PROFILE_DATASET_DESCRIPTION = """\
Profile a local .csv or .xlsx file and return the results as JSON.

Use this to understand a dataset before working with it: which columns it
has, their data types, how complete they are, basic statistics, and which
columns appear to contain personal or sensitive data (emails, phone
numbers, SSNs, credit cards, addresses, IP addresses and so on). The file is
read locally and is not modified.

Sample values are off by default. When requested, samples of columns
labeled as sensitive are replaced by "[masked: <LABEL>]", and in Lite mode
(detect_pii=false) every sample is "[hidden: Lite mode]"; raw sensitive
values are never returned.

Returns:
    A summary (file_name, file_format, mode, profiled_at, row_count,
    column_count, sheet_count) and a list of sheets (sheet_name is null for
    CSV), each with row_count, column_count, rows_profiled, sampled, note
    and columns. Each column has column_name, data_type, null_count,
    null_percent, unique_count, unique_ratio, rows_profiled, and when they
    apply data_label, data_label_confidence, min, max, mean, median,
    stddev and sample_values. Fields that don't apply are left out. Large
    tables are profiled on a random sample of rows; then sampled is true and
    counts describe the sample.
"""

EXPORT_PROFILE_DESCRIPTION = """\
Profile a local .csv or .xlsx file and save the profile as a file.

Use this when the user wants a profile they can open or share. The profile
has the same format as the input: a .csv input gives a .csv profile with
one row per column; an .xlsx input gives an .xlsx profile with a Summary
sheet, one profile sheet per input sheet and a Field guide sheet. Sample
values of sensitive columns are masked as "[masked: <LABEL>]" in the file,
and in Lite mode (detect_pii=false) all samples are "[hidden: Lite mode]".

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
    path = _input_path(file_path)
    profile = _run(lambda: _profile(path, detect_pii))
    return profile.to_dict(include_samples=include_samples)


@server.tool(
    description=EXPORT_PROFILE_DESCRIPTION,
    annotations=ToolAnnotations(
        title="Save a profile file next to a CSV or Excel file",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def export_profile(
    file_path: FilePath, output_path: OutputPath = None, detect_pii: DetectPii = True
) -> str:
    """Profile a file and save it (see EXPORT_PROFILE_DESCRIPTION)."""
    path = _input_path(file_path)
    target = _output_path(path, output_path)
    exported = _run(lambda: exporters.export_profile(_profile(path, detect_pii)))
    _run(lambda: target.write_bytes(exported.data))
    return str(target)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _profile(path: Path, detect_pii: bool) -> profiler.DatasetProfile:
    profiler.check_size(path.stat().st_size)  # before reading it into memory
    return profiler.profile_file(
        path.read_bytes(), path.name, detect_pii=detect_pii, get_labeler=get_labeler
    )


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
        raise ToolError(
            "Permission denied: the server isn't allowed to read or write that file."
        ) from None
    except OSError as exc:
        raise ToolError(
            f"Couldn't access the file: {exc.strerror or 'I/O error'}."
        ) from None
    except Exception as exc:
        logger.error("Profiling failed with %s", type(exc).__name__)
        raise ToolError(GENERIC_ERROR) from None


def _input_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if path.suffix.lower() not in profiler.SUPPORTED_EXTENSIONS:
        raise ToolError(
            f"Only .csv and .xlsx files are supported, and '{path.name}' is "
            "neither. Ask the user to save it as CSV or Excel Workbook (.xlsx)."
        )
    path = _run(path.resolve)
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    return path


def _output_path(input_path: Path, output_path: str | None) -> Path:
    suffix = input_path.suffix.lower()
    default_name = exporters.output_file_name(input_path.name)
    if output_path is None:
        target = input_path.parent / default_name
    else:
        target = Path(output_path).expanduser()
        if not target.is_absolute():
            target = input_path.parent / target
        target = _run(target.resolve)
        if target.is_dir():
            target = target / default_name
        elif target.suffix.lower() != suffix:
            raise ToolError(
                f"output_path must end in {suffix}, the same format as the input "
                "file (or be an existing folder)."
            )
    if target == input_path:
        raise ToolError("output_path can't be the input file itself.")
    if not target.parent.is_dir():
        raise ToolError(f"The folder {target.parent} doesn't exist.")
    if target.exists() and not target.name.lower().endswith(f"_profile{suffix}"):
        raise ToolError(
            f"{target} already exists and isn't a previous profile, so it won't be "
            "overwritten. Choose another output_path."
        )
    return target


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
