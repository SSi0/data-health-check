"""Masking rules for sample values of sensitive columns.

Shared by the web app and the MCP server so both apply exactly the same rule:
a column whose DataProfiler ``data_label`` names a sensitive type never shows
its raw sample values; ``[masked: <LABEL>]`` is shown instead.
"""

from __future__ import annotations

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


def mask_samples(samples: list[str], data_label: str | None) -> list[str]:
    """Return ``samples`` unchanged, or a single placeholder if sensitive."""
    if is_sensitive(data_label):
        return [mask_placeholder(data_label)]
    return list(samples)
