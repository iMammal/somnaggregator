"""Load manually entered sleep summaries into analysis-ready frames."""

from __future__ import annotations

import re
from datetime import date, datetime, time
from typing import Mapping, Sequence

import pandas as pd

from .schema import SleepTimeline


SUMMARY_COLUMNS = [
    "night_date",
    "device",
    "source",
    "sleep_start",
    "sleep_end",
    "total_sleep_minutes",
    "in_bed_minutes",
    "awake_minutes",
    "rem_minutes",
    "light_minutes",
    "deep_minutes",
    "hr_mean",
    "hrv_mean",
    "spo2_mean",
    "movement",
    "cpap_usage_minutes",
    "pressure_median",
    "leak_median",
    "event_count",
    "ahi",
    "notes",
]

_TIME_ONLY = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*([ap]\.?m\.?)?\s*$", re.IGNORECASE)


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clock_timestamp(night_date: object, value: object) -> pd.Timestamp:
    if _is_blank(value):
        return pd.NaT

    base = pd.to_datetime(night_date).normalize()
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, time):
        return pd.Timestamp.combine(base.date(), value)
    if isinstance(value, date):
        return pd.Timestamp(value)

    text = str(value).strip()
    if _TIME_ONLY.match(text):
        parsed_time = pd.to_datetime(text).time()
        return pd.Timestamp.combine(base.date(), parsed_time)
    return pd.to_datetime(text)


def parse_sleep_window(night_date: object, start: object, end: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a clock-time sleep window, allowing the end to cross midnight."""

    start_ts = _clock_timestamp(night_date, start)
    end_ts = _clock_timestamp(night_date, end)
    if pd.notna(start_ts) and pd.notna(end_ts) and end_ts <= start_ts:
        end_ts = end_ts + pd.Timedelta(days=1)
    return start_ts, end_ts


def load_manual_summaries(records: Sequence[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    """Create a clean manual-entry dataframe from dictionaries or an existing frame."""

    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    for column in SUMMARY_COLUMNS:
        if column not in df:
            df[column] = pd.NA

    if df.empty:
        return df[SUMMARY_COLUMNS]

    df["night_date"] = pd.to_datetime(df["night_date"]).dt.date
    for row_index, row in df.iterrows():
        start, end = parse_sleep_window(row["night_date"], row["sleep_start"], row["sleep_end"])
        df.at[row_index, "sleep_start"] = start
        df.at[row_index, "sleep_end"] = end

    numeric_columns = [
        "total_sleep_minutes",
        "in_bed_minutes",
        "awake_minutes",
        "rem_minutes",
        "light_minutes",
        "deep_minutes",
        "hr_mean",
        "hrv_mean",
        "spo2_mean",
        "movement",
        "cpap_usage_minutes",
        "pressure_median",
        "leak_median",
        "event_count",
        "ahi",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df[SUMMARY_COLUMNS].sort_values(["night_date", "device"]).reset_index(drop=True)


def summaries_to_timelines(summary_frame: pd.DataFrame) -> list[SleepTimeline]:
    """Convert manual summary rows into coarse one-segment timelines.

    This is intentionally simple: until true exports are loaded, a summary row
    becomes one interval marked "asleep" with the metrics that are available.
    """

    timelines: list[SleepTimeline] = []
    if summary_frame.empty:
        return timelines

    for _, row in summary_frame.iterrows():
        if pd.isna(row["sleep_start"]) or pd.isna(row["sleep_end"]):
            continue

        segment = pd.DataFrame(
            [
                {
                    "start": row["sleep_start"],
                    "end": row["sleep_end"],
                    "stage": "asleep",
                    "hr": row["hr_mean"],
                    "hrv": row["hrv_mean"],
                    "spo2": row["spo2_mean"],
                    "movement": row["movement"],
                    "pressure": row["pressure_median"],
                    "leak": row["leak_median"],
                    "event_count": row["event_count"],
                    "ahi": row["ahi"],
                    "cpap_usage": bool(pd.notna(row["cpap_usage_minutes"]) and row["cpap_usage_minutes"] > 0),
                    "event_flag": bool(pd.notna(row["event_count"]) and row["event_count"] > 0),
                    "source": row["source"],
                    "notes": row["notes"],
                }
            ]
        )
        timelines.append(
            SleepTimeline(
                device=str(row["device"]),
                night_date=str(row["night_date"]),
                segments=segment,
                source=None if pd.isna(row["source"]) else str(row["source"]),
            )
        )

    return timelines
