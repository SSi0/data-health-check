"""Masking rules for values copied from the data.

Shared by the web app and the MCP server so both apply exactly the same rules.
"Values copied from the data" are the sample values and the statistics that
can equal a single real value (min, max and median of number columns):

* in a column whose DataProfiler ``data_label`` names a sensitive type, they
  are replaced by ``[masked: <LABEL>]``;
* in Lite mode no labels exist, so no column can be shown to be safe: every
  sample, and min/max/median of every number column, is replaced by
  ``[hidden: Lite mode]``.

Mean and standard deviation are aggregates and are kept. Detection is only as
good as DataProfiler's labeler: a column it doesn't recognise (for example
names labelled UNKNOWN) is not masked.
"""

from __future__ import annotations

from typing import Any

LITE_MODE_PLACEHOLDER = "[hidden: Lite mode]"

# Statistics that can be equal to one real value in the column.
VALUE_STATISTICS: tuple[str, ...] = ("min", "max", "median")
NUMBER_TYPES = frozenset({"int", "float"})

SENSITIVE_LABELS: frozenset[str] = frozenset(
    {
        "ADDRESS",
        "BAN",
        "CREDIT_CARD",
        "DRIVERS_LICENSE",
        "EMAIL_ADDRESS",
        "PERSON",
        "PHONE_NUMBER",
        "SSN",
        "IPV4",
        "IPV6",
        "MAC_ADDRESS",
    }
)


def sensitive_labels(data_label: str | None) -> list[str]:
    """Return the sensitive labels contained in ``data_label``.

    DataProfiler joins near-tied labels with ``|`` (e.g. ``"UNKNOWN|ADDRESS"``),
    so every part is checked: one sensitive part is enough to mask the column.
    """
    if not data_label:
        return []
    parts = (part.strip().upper() for part in data_label.split("|"))
    return [part for part in parts if part in SENSITIVE_LABELS]


def is_sensitive(data_label: str | None) -> bool:
    """Return True if sample values for this label must be masked."""
    return bool(sensitive_labels(data_label))


def mask_placeholder(data_label: str | None) -> str:
    """Return the text shown instead of samples, e.g. ``[masked: SSN]``."""
    return f"[masked: {'|'.join(sensitive_labels(data_label))}]"


def mask_samples(
    samples: list[str], data_label: str | None, *, labels_checked: bool = True
) -> list[str]:
    """Return the samples that are safe to show.

    ``labels_checked`` is False in Lite mode, when the data labeler did not run:
    then every sample is hidden. Otherwise samples of sensitive columns are
    replaced by one placeholder and the rest are returned unchanged.
    """
    if not samples:
        return []
    if not labels_checked:
        return [LITE_MODE_PLACEHOLDER]
    if is_sensitive(data_label):
        return [mask_placeholder(data_label)]
    return list(samples)


def protect_column(column: dict[str, Any], *, labels_checked: bool) -> dict[str, Any]:
    """Return a copy of a profiled column with values copied from the data masked.

    ``column`` is one flattened column (``sample_values`` as a list).
    ``labels_checked`` is False in Lite mode, when the data labeler did not run.
    """
    label = column.get("data_label")
    protected = dict(column)
    protected["sample_values"] = mask_samples(
        column.get("sample_values") or [], label, labels_checked=labels_checked
    )
    if not labels_checked:
        if column.get("data_type") not in NUMBER_TYPES:
            return protected  # date min/max are not on the sensitive list
        placeholder = LITE_MODE_PLACEHOLDER
    elif is_sensitive(label):
        placeholder = mask_placeholder(label)
    else:
        return protected
    for key in VALUE_STATISTICS:
        if protected.get(key) is not None:
            protected[key] = placeholder
    return protected
