"""Data Health Check web app (Streamlit). UI only: the engine is in the package.

Run locally with ``streamlit run app.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from data_health_check import exporters, profiler
except ModuleNotFoundError:  # running from a checkout without `pip install .`
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_health_check import exporters, profiler

logger = logging.getLogger("data_health_check.app")

REPO_URL = "https://github.com/SSi0/data-health-check"
LOCAL_RUN_URL = f"{REPO_URL}#run-on-your-own-computer"
COMPANY_URL = "https://saltbridgesystems.com"

PRIVACY_NOTICE = (
    "Your file is processed on a hosted server, in memory only, and is not "
    "stored. Please don't upload confidential or regulated data. For sensitive "
    "data, you can run this app on your own computer "
    f"(see the [README]({LOCAL_RUN_URL}))."
)
GENERIC_ERROR = (
    "Something went wrong while profiling this file. Please check that it opens "
    "correctly in Excel, or try Lite mode (turn off sensitive data type "
    "detection in the sidebar)."
)
PII_TOGGLE = "Detect sensitive data types (PII)"
REVEAL_TOGGLE = "Show sensitive sample values"


@st.cache_resource(show_spinner=False)
def cached_labeler():
    """DataProfiler's labeler model, loaded once per server process."""
    return profiler.load_labeler()


def init_state() -> None:
    state = st.session_state
    state.setdefault("uploader_key", 0)
    if "pii_default" not in state:
        state.pii_available = profiler.pii_detection_available()
        state.pii_default = profiler.full_mode_fits()
        state.memory_available_mb = profiler.available_memory_mb()
        state.memory_needed_mb = profiler.estimate_memory_mb(
            profiler.MAX_FILE_BYTES, "csv", detect_pii=True
        )
        state[PII_TOGGLE] = state.pii_default
    state.setdefault(REVEAL_TOGGLE, False)


def sidebar() -> tuple[bool, bool]:
    state = st.session_state
    with st.sidebar:
        st.header("Settings")
        detect_pii = st.toggle(
            PII_TOGGLE,
            key=PII_TOGGLE,
            disabled=not state.pii_available,
            help=(
                "On (Full mode): a machine-learning model also works out what each "
                "column contains, such as emails, phone numbers, Social Security "
                "numbers or addresses. Off (Lite mode): types, completeness and "
                "statistics only; faster and uses much less memory."
            ),
        )
        if detect_pii:
            st.caption(
                "**Full mode.** The first check after the app starts loads the "
                "detection model, which can take a minute."
            )
        else:
            st.caption("**Lite mode.** No sensitive data detection.")
        if not state.pii_available:
            st.info(profiler.DETECTION_UNAVAILABLE_MESSAGE, icon=":material/info:")
        elif not state.pii_default:
            st.info(
                "Sensitive data detection is off by default because this server "
                f"has about {state.memory_available_mb:,.0f} MB of free memory and "
                f"detection needs about {state.memory_needed_mb:,.0f} MB for a "
                "large file. You can turn it on to try a small file, or "
                f"[run the app on your own computer]({LOCAL_RUN_URL}) for full "
                "detection.",
                icon=":material/memory:",
            )

        reveal = st.toggle(
            REVEAL_TOGGLE,
            key=REVEAL_TOGGLE,
            help=(
                "Off: example values in columns detected as sensitive are replaced "
                "by [masked: LABEL], and in Lite mode (which can't detect them) all "
                "example values are hidden, on screen and in the download. Turn on "
                "only if you need to see them."
            ),
        )
    return detect_pii, reveal


def intro() -> None:
    st.title("Data Health Check")
    st.markdown(
        "Get a quick health report for a spreadsheet: what kind of data each "
        "column holds, how complete it is, and whether it seems to contain "
        "personal information."
    )
    how, what = st.columns(2, gap="large")
    with how:
        st.subheader("How it works")
        st.markdown(
            f"1. **Drop your file** (.csv or .xlsx, up to {profiler.MAX_FILE_MB} MB) "
            "onto the box below, or click **Upload** to pick it.\n"
            "2. **Wait a moment** while it is checked. The first check after the "
            "app wakes up can take a minute or two.\n"
            "3. **Review the summary** and click **Download profile**."
        )
    with what:
        st.subheader("What you'll get")
        st.markdown(
            "- One row for **each column** in your file: its type (number, text, "
            "date), how many values are **missing**, how many are **unique**, "
            "basic statistics and a few examples.\n"
            "- With sensitive data detection on: what each column **appears to "
            "contain** (for example email addresses, phone numbers or Social "
            "Security numbers) and how sure the detector is.\n"
            "- A **download in the same format** as your file: a CSV file gives a "
            "CSV profile; an Excel file gives an Excel profile with a Summary, one "
            "sheet per sheet in your workbook, and a Field guide."
        )


def run_profile(uploaded, detect_pii: bool) -> None:
    """Profile the upload, keep only the result, and drop the file itself."""
    state = st.session_state
    state.pop("profile", None)
    state.pop("error", None)
    progress = st.progress(0.0, text="Starting…")

    def on_progress(fraction: float, message: str) -> None:
        progress.progress(min(max(fraction, 0.0), 1.0), text=message)

    try:
        state.profile = profiler.profile_file(
            uploaded.getvalue(),
            uploaded.name,
            detect_pii=detect_pii,
            get_labeler=cached_labeler,
            progress=on_progress,
        )
    except profiler.ProfilingError as exc:
        message = str(exc)
        if isinstance(exc, profiler.OutOfMemoryError) and detect_pii:
            message += f" ('{PII_TOGGLE}' is in the sidebar.)"
        state.error = message
    except Exception as exc:
        # Log the type only: messages can quote values from the file.
        logger.error("Profiling failed with %s", type(exc).__name__)
        state.error = GENERIC_ERROR
    finally:
        # A new key gives a fresh, empty uploader, so Streamlit releases the
        # uploaded bytes; only the profile (statistics + a few samples) is kept.
        state.uploader_key += 1
    st.rerun()


def display_frame(
    profile: profiler.DatasetProfile, sheet, reveal: bool
) -> pd.DataFrame:
    frame = pd.DataFrame(
        profile.column_rows(sheet, reveal_sensitive=reveal),
        columns=list(profile.column_fields()),
    )
    # min/max hold numbers for number columns and text for dates.
    for name in ("min", "max"):
        if frame[name].map(lambda v: isinstance(v, str)).any():
            frame[name] = frame[name].map(lambda v: "" if v is None else str(v))
    return frame


def show_results(
    profile: profiler.DatasetProfile, detect_pii: bool, reveal: bool
) -> None:
    st.success(
        f"**{profile.file_name}** has been profiled. The file itself has been "
        "discarded; only this summary is kept, until you close the page.",
        icon=":material/check_circle:",
    )
    if profile.detect_pii != detect_pii:
        st.caption(
            "You changed the sensitive data detection setting. Upload the file "
            "again to profile it with the new setting."
        )

    metrics = [("Rows", f"{profile.row_count:,}"), ("Columns", profile.column_count)]
    if profile.file_format == "xlsx":
        metrics.append(("Sheets", len(profile.sheets)))
    metrics.append(("Mode", "Full" if profile.detect_pii else "Lite"))
    for column, (label, value) in zip(st.columns(len(metrics)), metrics, strict=True):
        column.metric(label, value)

    exported = exporters.export_profile(profile, reveal_sensitive=reveal)
    st.download_button(
        f"Download profile ({exported.file_name})",
        data=exported.data,
        file_name=exported.file_name,
        mime=exported.mime_type,
        type="primary",
        icon=":material/download:",
        on_click="ignore",
    )
    if not reveal:
        hidden = (
            "Example values in columns detected as sensitive are masked"
            if profile.detect_pii
            else "Example values are hidden in Lite mode, because it can't tell "
            "which columns are sensitive"
        )
        st.caption(
            f"{hidden}, on screen and in the download. To show them, use "
            f"'{REVEAL_TOGGLE}' in the sidebar."
        )

    column_config = {
        name: st.column_config.Column(help=profiler.FIELD_GUIDE[name])
        for name in profile.column_fields()
    }
    for sheet in profile.sheets:
        if sheet.sheet_name is not None:
            st.subheader(f"Sheet: {sheet.sheet_name}")
        if sheet.note:
            st.caption(sheet.note)
        if sheet.columns:
            st.dataframe(
                display_frame(profile, sheet, reveal),
                hide_index=True,
                column_config=column_config,
                # Show every row (one per input column) up to a sensible cap.
                height=min(len(sheet.columns) + 1, 41) * 35 + 3,
                placeholder="",
            )


def field_guide(fields: tuple[str, ...]) -> None:
    with st.expander("What do the columns in the profile mean?"):
        st.table(
            pd.DataFrame(
                [(name, profiler.FIELD_GUIDE[name]) for name in fields],
                columns=["Field", "What it means"],
            ).set_index("Field")
        )


def footer() -> None:
    st.divider()
    st.caption(
        "Profiling powered by DataProfiler (Apache 2.0). Not affiliated with "
        f"Capital One. © 2026 [Saltbridge Systems LLC]({COMPANY_URL})."
    )


def main() -> None:
    st.set_page_config(
        page_title="Data Health Check",
        page_icon=":material/monitor_heart:",
        layout="wide",
    )
    init_state()
    detect_pii, reveal = sidebar()
    intro()

    st.info(PRIVACY_NOTICE, icon=":material/lock:")
    uploaded = st.file_uploader(
        "Drag and drop a .csv or .xlsx file here, or click Upload",
        key=f"upload-{st.session_state.uploader_key}",
        max_upload_size=profiler.MAX_FILE_MB,
        help=f"Up to {profiler.MAX_FILE_MB} MB. Only the first row is used as "
        "column names.",
    )
    if uploaded is not None:
        run_profile(uploaded, detect_pii)

    if "error" in st.session_state:
        st.error(st.session_state.error, icon=":material/error:")
    profile = st.session_state.get("profile")
    if profile is not None:
        show_results(profile, detect_pii, reveal)
        field_guide(profile.column_fields())
    else:
        field_guide(profiler.COLUMN_FIELDS)
    footer()


main()
