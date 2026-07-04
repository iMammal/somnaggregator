"""Core schemas and normalization helpers for SleepPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


STAGE_ORDER = ["awake", "rem", "light", "deep", "asleep", "no_data", "unknown"]
STAGE_LABELS = {
    "awake": "Awake",
    "rem": "REM",
    "light": "Light",
    "deep": "Deep",
    "asleep": "Asleep",
    "no_data": "No data",
    "unknown": "Unknown",
}
STAGE_COLORS = {
    "awake": "#f28e2b",
    "rem": "#59a14f",
    "light": "#4e79a7",
    "deep": "#1f3b73",
    "asleep": "#76b7b2",
    "no_data": "#d9d9d9",
    "unknown": "#b07aa1",
}

NUMERIC_COLUMNS = [
    "hr",
    "hrv",
    "spo2",
    "movement",
    "pressure",
    "leak",
    "event_count",
    "ahi",
]
FLAG_COLUMNS = ["cpap_usage", "event_flag"]
OPTIONAL_COLUMNS = NUMERIC_COLUMNS + FLAG_COLUMNS + ["source", "notes"]
TIMELINE_COLUMNS = [
    "night_date",
    "device",
    "start",
    "end",
    "stage",
    *OPTIONAL_COLUMNS,
]

SUMMARY_METRICS = [
    "night_date",
    "device",
    "total_sleep_minutes",
    "time_in_bed_minutes",
    "sleep_efficiency_pct",
    "sleep_score",
    "avg_hr_bpm",
    "min_hr_bpm",
    "avg_hrv_ms",
    "avg_spo2_pct",
    "min_spo2_pct",
    "respiratory_rate_bpm",
    "temperature_deviation_c",
    "readiness_score",
    "cpap_ahi",
    "cpap_usage_hours",
    "cpap_leak_rate",
    "cpap_pressure",
    "awake_minutes",
    "rem_minutes",
    "light_minutes",
    "deep_minutes",
    "max_hrv_ms",
    "hrv_balance_score",
    "breathing_label",
    "cpap_cai",
    "cpap_oai",
]
SUMMARY_VALUE_METRICS = [metric for metric in SUMMARY_METRICS if metric not in {"night_date", "device"}]
OBSERVATION_COLUMNS = [
    "night_date",
    "device",
    "metric",
    "value",
    "unit",
    "source_file",
    "extraction_method",
    "confidence",
    "notes",
]
NIGHTLY_SUMMARY_COLUMNS = SUMMARY_METRICS + [
    "source_files",
    "extraction_methods",
    "min_confidence",
    "notes",
]
PLOT_METRICS = [
    "total_sleep_minutes",
    "avg_hrv_ms",
    "avg_spo2_pct",
    "cpap_ahi",
]
OPTIONAL_DEVICE_METRICS = {
    "cpap": [
        "cpap_ahi",
        "cpap_usage_hours",
        "cpap_leak_rate",
        "cpap_pressure",
        "cpap_cai",
        "cpap_oai",
    ],
}
OPTIONAL_PLOT_METRICS = ["cpap_ahi"]
CANONICAL_METRIC_UNITS = {
    "total_sleep_minutes": "minutes",
    "time_in_bed_minutes": "minutes",
    "sleep_efficiency_pct": "pct",
    "sleep_score": "score",
    "avg_hr_bpm": "bpm",
    "min_hr_bpm": "bpm",
    "avg_hrv_ms": "ms",
    "avg_spo2_pct": "pct",
    "min_spo2_pct": "pct",
    "respiratory_rate_bpm": "breaths/min",
    "temperature_deviation_c": "C",
    "readiness_score": "score",
    "cpap_ahi": "events/hour",
    "cpap_usage_hours": "hours",
    "cpap_leak_rate": "L/min",
    "cpap_pressure": "cmH2O",
    "awake_minutes": "minutes",
    "rem_minutes": "minutes",
    "light_minutes": "minutes",
    "deep_minutes": "minutes",
    "max_hrv_ms": "ms",
    "hrv_balance_score": "score",
    "breathing_label": "label",
    "cpap_cai": "events/hour",
    "cpap_oai": "events/hour",
}
CANONICAL_METRICS = list(CANONICAL_METRIC_UNITS)

_METRIC_ALIASES = {
    "total_sleep": "total_sleep_minutes",
    "sleep": "total_sleep_minutes",
    "sleep_duration": "total_sleep_minutes",
    "asleep": "total_sleep_minutes",
    "time_asleep": "total_sleep_minutes",
    "total_sleep_minutes": "total_sleep_minutes",
    "time_in_bed": "time_in_bed_minutes",
    "in_bed": "time_in_bed_minutes",
    "time_in_bed_minutes": "time_in_bed_minutes",
    "sleep_efficiency": "sleep_efficiency_pct",
    "sleep_efficiency_pct": "sleep_efficiency_pct",
    "sleep_score": "sleep_score",
    "score": "sleep_score",
    "average_hr": "avg_hr_bpm",
    "avg_hr": "avg_hr_bpm",
    "average_heart_rate": "avg_hr_bpm",
    "avg_heart_rate": "avg_hr_bpm",
    "avg_hr_bpm": "avg_hr_bpm",
    "lowest_hr": "min_hr_bpm",
    "lowest_heart_rate": "min_hr_bpm",
    "minimum_hr": "min_hr_bpm",
    "minimum_heart_rate": "min_hr_bpm",
    "min_hr": "min_hr_bpm",
    "min_hr_bpm": "min_hr_bpm",
    "average_hrv": "avg_hrv_ms",
    "avg_hrv": "avg_hrv_ms",
    "avg_hrv_ms": "avg_hrv_ms",
    "max_hrv": "max_hrv_ms",
    "maximum_hrv": "max_hrv_ms",
    "max_hrv_ms": "max_hrv_ms",
    "hrv_balance": "hrv_balance_score",
    "hrv_balance_score": "hrv_balance_score",
    "oxygen_saturation": "avg_spo2_pct",
    "spo2": "avg_spo2_pct",
    "sp_o2": "avg_spo2_pct",
    "average_spo2": "avg_spo2_pct",
    "avg_spo2": "avg_spo2_pct",
    "avg_spo2_pct": "avg_spo2_pct",
    "average_oxygen": "avg_spo2_pct",
    "avg_oxygen": "avg_spo2_pct",
    "lowest_spo2": "min_spo2_pct",
    "minimum_spo2": "min_spo2_pct",
    "min_spo2": "min_spo2_pct",
    "min_spo2_pct": "min_spo2_pct",
    "lowest_oxygen": "min_spo2_pct",
    "respiratory_rate": "respiratory_rate_bpm",
    "respiration_rate": "respiratory_rate_bpm",
    "respiratory_rate_bpm": "respiratory_rate_bpm",
    "temperature_deviation": "temperature_deviation_c",
    "temp_deviation": "temperature_deviation_c",
    "temperature_deviation_c": "temperature_deviation_c",
    "readiness": "readiness_score",
    "readiness_score": "readiness_score",
    "ahi": "cpap_ahi",
    "cpap_ahi": "cpap_ahi",
    "events_hour": "cpap_ahi",
    "events_per_hour": "cpap_ahi",
    "usage": "cpap_usage_hours",
    "cpap_usage": "cpap_usage_hours",
    "mask_time": "cpap_usage_hours",
    "cpap_mask_minutes": "cpap_usage_hours",
    "cpap_usage_hours": "cpap_usage_hours",
    "mask_leak": "cpap_leak_rate",
    "leak": "cpap_leak_rate",
    "leak_rate": "cpap_leak_rate",
    "cpap_leak": "cpap_leak_rate",
    "cpap_leak_rate": "cpap_leak_rate",
    "pressure": "cpap_pressure",
    "pressure_95": "cpap_pressure",
    "95_pressure": "cpap_pressure",
    "cpap_pressure_95": "cpap_pressure",
    "cpap_pressure": "cpap_pressure",
    "awake": "awake_minutes",
    "wake": "awake_minutes",
    "awake_minutes": "awake_minutes",
    "rem": "rem_minutes",
    "rem_minutes": "rem_minutes",
    "light": "light_minutes",
    "core": "light_minutes",
    "light_minutes": "light_minutes",
    "deep": "deep_minutes",
    "deep_minutes": "deep_minutes",
    "breathing_label": "breathing_label",
    "cai": "cpap_cai",
    "cpap_cai": "cpap_cai",
    "oai": "cpap_oai",
    "cpap_oai": "cpap_oai",
}

_STAGE_ALIASES = {
    "wake": "awake",
    "awake": "awake",
    "waso": "awake",
    "rem": "rem",
    "r": "rem",
    "light": "light",
    "core": "light",
    "n1": "light",
    "n2": "light",
    "deep": "deep",
    "slow wave": "deep",
    "slow-wave": "deep",
    "sws": "deep",
    "n3": "deep",
    "asleep": "asleep",
    "sleep": "asleep",
    "cpap": "asleep",
    "no data": "no_data",
    "nodata": "no_data",
    "missing": "no_data",
    "unknown": "unknown",
    "": "unknown",
}


def normalize_stage(value: object) -> str:
    """Map device-specific stage labels into a small common vocabulary."""

    if value is None or pd.isna(value):
        return "unknown"
    key = str(value).strip().lower().replace("_", " ")
    return _STAGE_ALIASES.get(key, key if key in STAGE_ORDER else "unknown")


def ensure_timeline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated timeline frame with common columns and dtypes."""

    required = {"night_date", "device", "start", "end", "stage"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Timeline frame is missing required columns: {sorted(missing)}")

    df = frame.copy()
    df["night_date"] = pd.to_datetime(df["night_date"]).dt.date
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["stage"] = df["stage"].map(normalize_stage)

    bad_duration = df["end"].notna() & df["start"].notna() & (df["end"] <= df["start"])
    if bad_duration.any():
        rows = df.index[bad_duration].tolist()
        raise ValueError(f"Timeline rows must have end after start; bad rows: {rows}")

    for column in OPTIONAL_COLUMNS:
        if column not in df:
            df[column] = pd.NA

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in FLAG_COLUMNS:
        df[column] = df[column].fillna(False).astype(bool)

    return df[TIMELINE_COLUMNS].sort_values(["night_date", "device", "start"]).reset_index(drop=True)


@dataclass
class SleepTimeline:
    """A single device's staged or metric timeline for one sleep night."""

    device: str
    night_date: str
    segments: pd.DataFrame
    source: str | None = None
    notes: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return normalized timeline rows with device and night metadata."""

        df = self.segments.copy()
        df["device"] = self.device
        df["night_date"] = self.night_date
        if self.source is not None and "source" not in df:
            df["source"] = self.source
        if self.notes is not None and "notes" not in df:
            df["notes"] = self.notes
        return ensure_timeline_frame(df)


def concat_timeline_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate normalized timeline frames, returning an empty schema if needed."""

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    return ensure_timeline_frame(pd.concat(frames, ignore_index=True))


def metric_key(value: object) -> str:
    """Normalize a raw metric label into a lookup-friendly key."""

    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("%", " pct ")
    text = text.replace("/", " per ")
    text = text.replace("+", " ")
    text = text.replace("-", " ")
    return "_".join(part for part in re_split_metric_label(text) if part)


def re_split_metric_label(text: str) -> list[str]:
    """Split a metric label without importing regex globally for older notebook reloads."""

    import re

    return re.split(r"[^a-z0-9]+", text)


def normalize_metric_name(value: object) -> str:
    """Map raw or legacy metric labels to stable canonical metric names."""

    key = metric_key(value)
    if key in _METRIC_ALIASES:
        return _METRIC_ALIASES[key]
    if key in CANONICAL_METRIC_UNITS:
        return key
    return str(value).strip() if value is not None and not pd.isna(value) else ""


def canonical_unit_for_metric(metric: object, fallback: object = pd.NA) -> object:
    """Return the canonical unit for a metric when known."""

    return CANONICAL_METRIC_UNITS.get(normalize_metric_name(metric), fallback)


def normalize_metric_value(metric: object, value: object, unit: object = None) -> object:
    """Convert legacy metric values to canonical units when needed."""

    if value is None or pd.isna(value):
        return pd.NA

    canonical = normalize_metric_name(metric)
    source_key = metric_key(metric)
    unit_key = metric_key(unit)

    if canonical == "cpap_usage_hours":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        if source_key == "cpap_mask_minutes" or unit_key in {"minute", "minutes", "min", "mins"}:
            hours = numeric / 60
            return round(hours, 3)
    return value


def empty_observations_frame() -> pd.DataFrame:
    """Return an empty long-form observation table."""

    return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def ensure_observations_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized long-form observation table with provenance columns."""

    df = frame.copy()
    if "night_date" not in df and "date" in df:
        df["night_date"] = df["date"]
    for column in OBSERVATION_COLUMNS:
        if column not in df:
            df[column] = pd.NA

    if df.empty:
        return df[OBSERVATION_COLUMNS]

    parsed_dates = pd.to_datetime(df["night_date"], errors="coerce")
    df["night_date"] = parsed_dates.dt.date.where(parsed_dates.notna(), df["night_date"])
    blank_date = df["night_date"].isna() | (df["night_date"].astype(str).str.strip() == "")
    df.loc[blank_date, "night_date"] = pd.NA

    original_metric = df["metric"].copy()
    original_unit = df["unit"].copy()
    df["metric"] = df["metric"].map(normalize_metric_name)
    df["value"] = [
        normalize_metric_value(metric, value, unit)
        for metric, value, unit in zip(original_metric, df["value"], original_unit)
    ]
    df["unit"] = [
        canonical_unit_for_metric(metric, unit)
        for metric, unit in zip(df["metric"], original_unit)
    ]
    df["device"] = df["device"].astype(str)
    df["source_file"] = df["source_file"].astype(str)
    df["extraction_method"] = df["extraction_method"].fillna("manual").astype(str)
    df["confidence"] = df["confidence"].fillna("low").astype(str)
    return df[OBSERVATION_COLUMNS].reset_index(drop=True)


def normalize_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy summary columns to canonical metric columns."""

    df = frame.copy()
    if "night_date" not in df and "date" in df:
        df = df.rename(columns={"date": "night_date"})

    rename_map = {}
    for column in df.columns:
        if column in {"night_date", "device", "source_files", "extraction_methods", "min_confidence", "notes"}:
            continue
        canonical = normalize_metric_name(column)
        if canonical and canonical != column:
            rename_map[column] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        deduped = pd.DataFrame(index=df.index)
        for column in dict.fromkeys(df.columns):
            values = df.loc[:, column]
            if isinstance(values, pd.DataFrame):
                deduped[column] = values.bfill(axis=1).iloc[:, 0]
            else:
                deduped[column] = values
        df = deduped

    for column in NIGHTLY_SUMMARY_COLUMNS:
        if column not in df:
            df[column] = pd.NA
    return df[NIGHTLY_SUMMARY_COLUMNS]
