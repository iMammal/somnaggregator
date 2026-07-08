"""Muse sleep EEG screenshot/PDF summary extraction."""

from __future__ import annotations

import re
from pathlib import Path

from .common import observation, parse_duration_to_minutes, parse_wellness_text, read_source_text, source_file_label


def parse_muse_text(
    text: str,
    *,
    source_file: str,
    device: str = "Muse",
    extraction_method: str = "parsed_text",
    confidence: str = "high",
    notes: str = "",
    page: int | None = None,
    **kwargs,
) -> list[dict[str, object]]:
    """Parse Muse sleep summary text conservatively."""

    rows = parse_wellness_text(
        text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        page=page,
    )
    return _augment_muse_rows(
        rows,
        text=text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        page=page,
    )


def extract_file(path: str | Path, device: str | None = None) -> list[dict[str, object]]:
    """Extract Muse observations from one screenshot or PDF."""

    source_path = Path(path)
    source = read_source_text(source_path)
    return parse_muse_text(
        source.text,
        source_file=source_file_label(source_path),
        device=device or "Muse",
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )


def diagnostic_summary(text: str) -> dict[str, str]:
    """Return a conservative explanation for Muse coverage in the extraction report."""

    lines = text.splitlines()
    snippets = {
        "session": _line_window(lines, ["sleep session", "time in bed", "time asleep"]),
        "stages": _line_window(lines, ["awake", "rem", "light", "deep"]),
        "heart_rate": _line_window(lines, ["heart rate", "bpm"]),
        "respiratory_rate": _line_window(lines, ["respiratory rate", "breaths/min", "times/min"]),
    }
    if not _clear_muse_labels_present(text):
        snippets["reason"] = "rotated image-only PDF; OCR text is too sparse and contains no clear Muse session/stage labels."
    return snippets


def _augment_muse_rows(
    rows: list[dict[str, object]],
    *,
    text: str,
    device: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
    page: int | None,
) -> list[dict[str, object]]:
    """Add Muse-specific rows only when the labels are clear."""

    row_map = {str(row.get("metric")): row for row in rows}
    date = rows[0].get("night_date") if rows else None

    sleep_score = _extract_sleep_score(text)
    if sleep_score is not None and "sleep_score" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=date,
            device=device,
            metric="sleep_score",
            value=sleep_score,
            unit="score",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
        )

    total_sleep = _extract_duration_after_label(text, r"(?:time\s+asleep|sleep\s+time)")
    if total_sleep is not None and "total_sleep_minutes" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=date,
            device=device,
            metric="total_sleep_minutes",
            value=total_sleep,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
        )

    time_in_bed = _extract_duration_after_label(text, r"time\s+in\s+bed")
    if time_in_bed is not None and "time_in_bed_minutes" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=date,
            device=device,
            metric="time_in_bed_minutes",
            value=time_in_bed,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
        )

    for metric, label in [
        ("awake_minutes", r"awake"),
        ("rem_minutes", r"rem"),
        ("light_minutes", r"light"),
        ("deep_minutes", r"deep"),
    ]:
        value = _extract_duration_after_label(text, label)
        if value is not None and metric not in row_map:
            _set_metric(
                rows,
                row_map,
                date=date,
                device=device,
                metric=metric,
                value=value,
                unit="minutes",
                source_file=source_file,
                extraction_method=extraction_method,
                confidence=confidence,
                notes=notes,
            )

    avg_hr = _extract_avg_hr_bpm(text)
    if avg_hr is not None and "avg_hr_bpm" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=date,
            device=device,
            metric="avg_hr_bpm",
            value=avg_hr,
            unit="bpm",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
        )

    respiratory_rate = _extract_respiratory_rate(text)
    if respiratory_rate is not None and "respiratory_rate_bpm" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=date,
            device=device,
            metric="respiratory_rate_bpm",
            value=respiratory_rate,
            unit="breaths/min",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
        )

    return rows


def _set_metric(
    rows: list[dict[str, object]],
    row_map: dict[str, dict[str, object]],
    *,
    date: str | None,
    device: str,
    metric: str,
    value: float | int,
    unit: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
) -> None:
    if float(value).is_integer():
        value_to_store: float | int = int(value)
    else:
        value_to_store = round(float(value), 3)

    current = row_map.get(metric)
    if current is not None:
        current["value"] = value_to_store
        current["unit"] = unit
        return

    row = observation(
        date=date if isinstance(date, str) or date is None else str(date),
        device=device,
        metric=metric,
        value=value_to_store,
        unit=unit,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
    )
    rows.append(row)
    row_map[metric] = row


def _extract_sleep_score(text: str) -> int | None:
    match = re.search(r"\b(?:sleep\s+session|sleep\s+score)\s*(?P<score>\d{1,3})\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group("score"))
    return None


def _extract_duration_after_label(text: str, label_pattern: str) -> int | None:
    match = re.search(
        rf"{label_pattern}\s*[:\-]?\s*(?P<duration>\d+(?:\.\d+)?\s*(?:hours?|hrs?|h)(?:\s*,?\s*\d+(?:\.\d+)?\s*(?:minutes?|mins?|m))?(?:\s*,?\s*\d+(?:\.\d+)?\s*(?:seconds?|secs?|s))?|\d+(?:\.\d+)?\s*(?:minutes?|mins?|m))",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return parse_duration_to_minutes(match.group("duration"))


def _extract_avg_hr_bpm(text: str) -> float | None:
    match = re.search(r"\baverage\s*[:\-]?\s*(?P<value>\d{2,3})\s*bpm\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group("value"))


def _extract_respiratory_rate(text: str) -> float | None:
    match = re.search(r"\baverage\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*times/min\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group("value"))


def _line_window(lines: list[str], keywords: list[str], radius: int = 2) -> str:
    for index, line in enumerate(lines):
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            return " | ".join(line.strip() for line in lines[start:end] if line.strip())
    return "(not found)"


def _clear_muse_labels_present(text: str) -> bool:
    return bool(
        re.search(r"\bsleep\s+session\b", text, flags=re.IGNORECASE)
        or re.search(r"\btime\s+in\s+bed\b", text, flags=re.IGNORECASE)
        or re.search(r"\btime\s+asleep\b", text, flags=re.IGNORECASE)
        or re.search(r"\bawake\b", text, flags=re.IGNORECASE)
        or re.search(r"\brem\b", text, flags=re.IGNORECASE)
        or re.search(r"\blight\b", text, flags=re.IGNORECASE)
        or re.search(r"\bdeep\b", text, flags=re.IGNORECASE)
    )
