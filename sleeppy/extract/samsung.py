"""Samsung Health / SleepWatch screenshot/PDF summary extraction."""

from __future__ import annotations

import re
from pathlib import Path

from .common import observation, parse_duration_to_minutes, parse_wellness_text, read_source_text, source_file_label


def parse_samsung_text(
    text: str,
    *,
    source_file: str,
    device: str = "Samsung Watch / SleepWatch",
    extraction_method: str = "parsed_text",
    confidence: str = "high",
    notes: str = "",
    page: int | None = None,
    **kwargs,
) -> list[dict[str, object]]:
    """Parse Samsung Health or SleepWatch sleep summary text."""

    rows = parse_wellness_text(
        text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        page=page,
    )
    return _augment_samsung_rows(
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
    """Extract Samsung/SleepWatch observations from one screenshot or PDF."""

    source_path = Path(path)
    source = read_source_text(source_path)
    return parse_samsung_text(
        source.text,
        source_file=source_file_label(source_path),
        device=device or "Samsung Watch / SleepWatch",
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )


def diagnostic_summary(text: str) -> dict[str, str]:
    """Return snippets for the Samsung layout so the extraction report can explain coverage."""

    lines = text.splitlines()
    stage_block = _stage_block_snippet(lines)
    snippets = {
        "sleep_duration": _line_window(lines, ["sleep time", "actual sleep time", "time in bed", "time asleep"]),
        "stages": stage_block,
        "blood_oxygen": _line_window(lines, ["blood oxygen", "spo2"]),
        "heart_rate": _line_window(lines, ["55 bpm", "average: 55 bpm", "age: 55 bpm"], radius=4),
        "respiratory_rate": _line_window(lines, ["respiratory rate", "times/min"]),
    }
    if snippets["heart_rate"] == "(not found)":
        snippets["heart_rate"] = _line_window(lines, ["heart rate", "bpm"], radius=6)
    if snippets["sleep_duration"] == "(not found)" or "over last 7 days" in snippets["sleep_duration"].lower():
        snippets["sleep_duration"] = stage_block
    if not _stage_block_present(lines) and not _has_clear_vitals(text):
        snippets["reason"] = "OCR text is too sparse for conservative Samsung sleep extraction."
    return snippets


def _augment_samsung_rows(
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
    """Add Samsung-specific metrics that the shared parser does not recover."""

    row_map = {str(row.get("metric")): row for row in rows}
    lines = text.splitlines()
    stage_values = _extract_stage_durations(lines)
    stage_note = _stage_note(lines)

    if stage_values is not None:
        awake_minutes, rem_minutes, light_minutes, deep_minutes = stage_values
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="awake_minutes",
            value=awake_minutes,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes if not stage_note else f"{notes}; {stage_note}",
        )
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="rem_minutes",
            value=rem_minutes,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes if not stage_note else f"{notes}; {stage_note}",
        )
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="light_minutes",
            value=light_minutes,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes if not stage_note else f"{notes}; {stage_note}",
        )
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="deep_minutes",
            value=deep_minutes,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes if not stage_note else f"{notes}; {stage_note}",
        )

        total_sleep_minutes = rem_minutes + light_minutes + deep_minutes
        time_in_bed_minutes = awake_minutes + total_sleep_minutes
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="total_sleep_minutes",
            value=total_sleep_minutes,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes if not stage_note else f"{notes}; {stage_note}",
        )
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="time_in_bed_minutes",
            value=time_in_bed_minutes,
            unit="minutes",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes if not stage_note else f"{notes}; {stage_note}",
        )

    sleep_score = _extract_sleep_score(text)
    if sleep_score is not None and "sleep_score" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="sleep_score",
            value=sleep_score,
            unit="score",
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
        )

    avg_hr_bpm = _extract_avg_hr_bpm(text)
    if avg_hr_bpm is not None and "avg_hr_bpm" not in row_map:
        _set_metric(
            rows,
            row_map,
            date=rows[0].get("night_date") if rows else None,
            device=device,
            metric="avg_hr_bpm",
            value=avg_hr_bpm,
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
            date=rows[0].get("night_date") if rows else None,
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
        if notes:
            current["notes"] = _merge_notes(str(current.get("notes", "")), notes)
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


def _extract_stage_durations(lines: list[str]) -> tuple[int, int, int, int] | None:
    """Return awake/rem/light/deep stage durations from the Samsung stage block."""

    start_index = None
    for index, line in enumerate(lines):
        lower = line.lower()
        if "awake" in lower and "%" in lower:
            start_index = index
            break
    if start_index is None:
        return None

    candidates: list[int] = []
    for line in lines[start_index:]:
        lower = line.lower()
        if "sleep time over last 7 days" in lower:
            break
        for token in re.findall(r"\b(?:\d+\s*h\s*\d+\s*m|\d+\s*h|\d+\s*m)\b", line, flags=re.IGNORECASE):
            minutes = parse_duration_to_minutes(token)
            if minutes is not None:
                candidates.append(minutes)
        if len(candidates) >= 4:
            break

    if len(candidates) < 4:
        return None
    return candidates[0], candidates[1], candidates[2], candidates[3]


def _stage_note(lines: list[str]) -> str:
    snippet = _stage_block_snippet(lines)
    if snippet == "(not found)":
        return ""
    return f"Samsung stage block: {snippet}"


def _extract_sleep_score(text: str) -> int | None:
    match = re.search(r"\b(?P<score>\d{1,3})\s+attention\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group("score"))
    return None


def _extract_avg_hr_bpm(text: str) -> float | None:
    match = re.search(r"\b(?:average|age)\s*[:\-]?\s*(?P<value>\d{2,3})\s*bpm\b", text, flags=re.IGNORECASE)
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


def _stage_block_snippet(lines: list[str]) -> str:
    start_index = None
    end_index = None
    for index, line in enumerate(lines):
        lower = line.lower()
        if start_index is None and "awake" in lower:
            start_index = index
        if start_index is not None and "sleep time over last 7 days" in lower:
            end_index = index
            break
    if start_index is None:
        return "(not found)"
    if end_index is None:
        end_index = min(len(lines), start_index + 25)
    return " | ".join(line.strip() for line in lines[start_index:end_index] if line.strip())


def _stage_block_present(lines: list[str]) -> bool:
    return any("awake" in line.lower() and "%" in line.lower() for line in lines)


def _has_clear_vitals(text: str) -> bool:
    return bool(
        re.search(r"\baverage\s*[:\-]?\s*\d{2,3}\s*bpm\b", text, flags=re.IGNORECASE)
        or re.search(r"\baverage\s*[:\-]?\s*\d+(?:\.\d+)?\s*times/min\b", text, flags=re.IGNORECASE)
        or re.search(r"\bblood oxygen\b", text, flags=re.IGNORECASE)
    )
