"""Utility helpers shared across NLP processing modules."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, List

import pandas


def to_str(val: Any) -> str:
    """Return a trimmed string representation for any value."""
    try:
        return str(val).strip()
    except Exception:
        return ""


def is_placeholder(text: str) -> bool:
    """Check if the provided text should be treated as a placeholder/empty value."""
    s = to_str(text).lower()
    if not s:
        return True
    if s in {"n/a", "na", "none", "null", "unknown", "unk", "-", "--"}:
        return True
    # Match patterns like [provide date], [select regimen], etc.
    if re.fullmatch(r"\[[^\]]+\]", s or ""):
        return True
    return False


def is_invalid_value(param_type: str, value: Any) -> bool:
    """Return True if the value is considered invalid for the given parameter type."""
    if value is None:
        return True
    try:
        if pandas.isna(value):
            return True
    except Exception:
        pass
    s = to_str(value)
    if not s:
        return True
    if is_placeholder(s):
        return True
    if param_type == "DATE":
        if not re.search(r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}", s):
            return True
    return False


def is_duplicate_row(
    excel_data: pandas.DataFrame,
    staged_rows: List[dict[str, Any]],
    patient_id: str,
    core_variable: str,
    value: Any,
    date_ref: str,
) -> bool:
    """Check if a row already exists in excel_data or staged rows."""
    if "patient_id" in excel_data.columns and not excel_data.empty:
        existing_rows = excel_data[excel_data["patient_id"] == patient_id]
        if not existing_rows.empty:
            dup = existing_rows[
                (existing_rows["core_variable"] == core_variable)
                & (existing_rows["value"].astype(str) == to_str(value))
                & (existing_rows["date_ref"].astype(str) == to_str(date_ref))
            ]
            if not dup.empty:
                return True
    for row in staged_rows:
        if (
            row.get("patient_id") == patient_id
            and row.get("core_variable") == core_variable
            and to_str(row.get("value")) == to_str(value)
            and to_str(row.get("date_ref")) == to_str(date_ref)
        ):
            return True
    return False


def parse_date_any(s: str) -> datetime | None:
    """Parse DD/MM/YYYY or YYYY-MM-DD formats; return None if not parseable."""
    if not s:
        return None
    s = to_str(s)
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def first_group(match: Any) -> Any:
    """Return the first element of a regex match tuple or the raw match."""
    if isinstance(match, tuple) and len(match) > 0:
        return match[0]
    return match
