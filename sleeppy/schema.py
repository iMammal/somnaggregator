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
    "date",
    "device",
    "sleep_score",
    "total_sleep_minutes",
    "time_in_bed_minutes",
    "sleep_efficiency_pct",
    "awake_minutes",
    "rem_minutes",
    "light_minutes",
    "deep_minutes",
    "lowest_hr",
    "avg_hr",
    "avg_hrv",
    "max_hrv",
    "avg_spo2",
    "respiratory_rate",
    "breathing_label",
    "cpap_mask_minutes",
    "cpap_ahi",
    "cpap_cai",
    "cpap_oai",
    "cpap_pressure_95",
    "cpap_leak",
]
SUMMARY_VALUE_METRICS = [metric for metric in SUMMARY_METRICS if metric not in {"date", "device"}]
OBSERVATION_COLUMNS = [
    "date",
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


def empty_observations_frame() -> pd.DataFrame:
    """Return an empty long-form observation table."""

    return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def ensure_observations_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized long-form observation table with provenance columns."""

    df = frame.copy()
    for column in OBSERVATION_COLUMNS:
        if column not in df:
            df[column] = pd.NA

    if df.empty:
        return df[OBSERVATION_COLUMNS]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["metric"] = df["metric"].astype(str)
    df["device"] = df["device"].astype(str)
    df["source_file"] = df["source_file"].astype(str)
    df["extraction_method"] = df["extraction_method"].fillna("manual").astype(str)
    df["confidence"] = df["confidence"].fillna("low").astype(str)
    return df[OBSERVATION_COLUMNS].reset_index(drop=True)
