"""Oura screenshot/PDF summary extraction."""

from __future__ import annotations

import re
from pathlib import Path

from .common import infer_device_from_path, observation, parse_duration_to_minutes, parse_wellness_text, read_source_text, source_file_label


def parse_oura_text(
    text: str,
    *,
    source_file: str,
    device: str = "Oura Ring",
    extraction_method: str = "parsed_text",
    confidence: str = "high",
    notes: str = "",
    page: int | None = None,
    **kwargs,
) -> list[dict[str, object]]:
    """Parse Oura sleep summary text into long-form observations."""

    rows = parse_wellness_text(
        text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        page=page,
    )
    return _normalize_oura_rows(
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
    """Extract Oura observations from one screenshot or PDF."""

    source_path = Path(path)
    source = read_source_text(source_path)
    inferred_device = device or infer_device_from_path(source_path)
    if inferred_device == "Unknown device":
        inferred_device = "Oura Ring"
    return parse_oura_text(
        source.text,
        source_file=source_file_label(source_path),
        device=inferred_device,
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )


def _normalize_oura_rows(
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
    """Apply Oura-specific consistency corrections to parsed rows."""

    if not rows:
        return rows

    filtered_rows = []
    explicit_zero_score = _contains_explicit_zero_sleep_score(text)
    for row in rows:
        if row.get("metric") == "sleep_score" and _numeric_value(row.get("value")) == 0 and not explicit_zero_score:
            continue
        filtered_rows.append(row)

    rows = filtered_rows

    def row_for(metric: str) -> dict[str, object] | None:
        for row in rows:
            if row.get("metric") == metric:
                return row
        return None

    def set_metric(metric: str, value: float, unit: str) -> None:
        current = row_for(metric)
        if float(value).is_integer():
            value_to_store: float | int = int(value)
        else:
            value_to_store = round(value, 3)
        if current is not None:
            current["value"] = value_to_store
            current["unit"] = unit
            return

        date = rows[0].get("night_date") if rows else None
        rows.append(
            observation(
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
        )

    def set_if_changed(metric: str, candidate: float, unit: str, tolerance: float = 10.0) -> None:
        current = row_for(metric)
        current_value = _numeric_value(current.get("value")) if current is not None else None
        if current_value is None or abs(current_value - candidate) > tolerance:
            set_metric(metric, candidate, unit)

    total_sleep = _numeric_value(row_for("total_sleep_minutes").get("value")) if row_for("total_sleep_minutes") else None
    time_in_bed = _numeric_value(row_for("time_in_bed_minutes").get("value")) if row_for("time_in_bed_minutes") else None
    efficiency = _numeric_value(row_for("sleep_efficiency_pct").get("value")) if row_for("sleep_efficiency_pct") else None
    awake = _numeric_value(row_for("awake_minutes").get("value")) if row_for("awake_minutes") else None
    rem = _numeric_value(row_for("rem_minutes").get("value")) if row_for("rem_minutes") else None
    deep = _numeric_value(row_for("deep_minutes").get("value")) if row_for("deep_minutes") else None
    light = _numeric_value(row_for("light_minutes").get("value")) if row_for("light_minutes") else None

    if total_sleep is not None and efficiency is not None and efficiency > 0:
        candidate_tib = round(total_sleep * 100.0 / efficiency)
        set_if_changed("time_in_bed_minutes", candidate_tib, "minutes")
        time_in_bed = _numeric_value(row_for("time_in_bed_minutes").get("value")) if row_for("time_in_bed_minutes") else time_in_bed

    if time_in_bed is None:
        total_duration_minutes = _extract_total_duration_minutes(text)
        if total_duration_minutes is not None:
            set_metric("time_in_bed_minutes", total_duration_minutes, "minutes")
            time_in_bed = total_duration_minutes

    if time_in_bed is not None and awake is not None:
        candidate_sleep = time_in_bed - awake
        if candidate_sleep > 0:
            set_if_changed("total_sleep_minutes", candidate_sleep, "minutes")
            total_sleep = _numeric_value(row_for("total_sleep_minutes").get("value")) if row_for("total_sleep_minutes") else total_sleep

    if time_in_bed is not None and total_sleep is not None and awake is None:
        candidate_awake = time_in_bed - total_sleep
        if candidate_awake >= 0:
            set_metric("awake_minutes", candidate_awake, "minutes")
            awake = candidate_awake

    if time_in_bed is not None:
        candidate_light = None
        if awake is not None and rem is not None and deep is not None:
            candidate_light = time_in_bed - awake - rem - deep
        elif total_sleep is not None and rem is not None and deep is not None:
            candidate_light = total_sleep - rem - deep
        if candidate_light is not None and candidate_light >= 0:
            set_if_changed("light_minutes", candidate_light, "minutes")
            light = _numeric_value(row_for("light_minutes").get("value")) if row_for("light_minutes") else light

    total_sleep = _numeric_value(row_for("total_sleep_minutes").get("value")) if row_for("total_sleep_minutes") else total_sleep
    time_in_bed = _numeric_value(row_for("time_in_bed_minutes").get("value")) if row_for("time_in_bed_minutes") else time_in_bed
    if time_in_bed is not None and total_sleep is not None and time_in_bed > 0:
        candidate_efficiency = round(total_sleep * 100.0 / time_in_bed)
        current_efficiency = _numeric_value(row_for("sleep_efficiency_pct").get("value")) if row_for("sleep_efficiency_pct") else None
        if current_efficiency is None or abs(current_efficiency - candidate_efficiency) > 2:
            set_metric("sleep_efficiency_pct", candidate_efficiency, "pct")

    return rows


def _extract_total_duration_minutes(text: str) -> int | None:
    """Extract Oura's total duration line when it is available."""

    match = re.search(
        r"total\s+duration\s*[:\-]?\s*(?P<duration>\d+(?:\.\d+)?\s*(?:hours?|hrs?|h)(?:\s*,?\s*\d+(?:\.\d+)?\s*(?:minutes?|mins?|m))?(?:\s*,?\s*\d+(?:\.\d+)?\s*(?:seconds?|secs?|s))?|\d+(?:\.\d+)?\s*(?:minutes?|mins?|m))",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return parse_duration_to_minutes(match.group("duration"))


def _numeric_value(value: object) -> float | None:
    """Return a float if a row value is numeric, otherwise None."""

    if value is None:
        return None
    try:
        if value != value:  # NaN check without importing pandas here.
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains_explicit_zero_sleep_score(text: str) -> bool:
    """Return True when the OCR text clearly contains an explicit zero sleep score."""

    return bool(re.search(r"\bsleep\s+score\b\s*[:\-]?\s*0\b", text, flags=re.IGNORECASE))
