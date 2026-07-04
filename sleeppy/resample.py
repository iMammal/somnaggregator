"""Resample timeline segments into fixed clock-time bins."""

from __future__ import annotations

import pandas as pd

from .schema import FLAG_COLUMNS, NUMERIC_COLUMNS, TIMELINE_COLUMNS, ensure_timeline_frame
from .timeline import align_timelines


def _empty_binned_frame() -> pd.DataFrame:
    columns = [
        "night_date",
        "device",
        "bin_start",
        "bin_end",
        "stage",
        "duration_minutes",
        "coverage_fraction",
        "cpap_usage_fraction",
        *NUMERIC_COLUMNS,
        *FLAG_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def resample_timelines(timelines_or_frame, freq: str = "15min") -> pd.DataFrame:
    """Resample aligned timelines into fixed-width bins.

    Stage values are selected by greatest overlap within each bin. Numeric
    metrics use overlap-weighted averages; flags use "any overlap".
    """

    if isinstance(timelines_or_frame, pd.DataFrame):
        frame = timelines_or_frame
    else:
        frame = align_timelines(timelines_or_frame)

    if frame.empty:
        return _empty_binned_frame()

    timeline = ensure_timeline_frame(frame[[column for column in TIMELINE_COLUMNS if column in frame.columns]])
    bin_delta = pd.Timedelta(freq)
    rows: list[dict[str, object]] = []

    for (night_date, device), group in timeline.groupby(["night_date", "device"], sort=True):
        first_bin = group["start"].min().floor(freq)
        last_bin = group["end"].max().ceil(freq)
        bin_starts = pd.date_range(first_bin, last_bin - bin_delta, freq=freq)

        for bin_start in bin_starts:
            bin_end = bin_start + bin_delta
            overlaps = group[(group["end"] > bin_start) & (group["start"] < bin_end)].copy()
            if overlaps.empty:
                rows.append(
                    {
                        "night_date": night_date,
                        "device": device,
                        "bin_start": bin_start,
                        "bin_end": bin_end,
                        "stage": "no_data",
                        "duration_minutes": bin_delta.total_seconds() / 60,
                        "coverage_fraction": 0.0,
                        "cpap_usage_fraction": 0.0,
                        **{column: pd.NA for column in NUMERIC_COLUMNS},
                        **{column: False for column in FLAG_COLUMNS},
                    }
                )
                continue

            overlap_start = overlaps["start"].where(overlaps["start"] > bin_start, bin_start)
            overlap_end = overlaps["end"].where(overlaps["end"] < bin_end, bin_end)
            overlaps["overlap_seconds"] = (overlap_end - overlap_start).dt.total_seconds()
            stage_seconds = overlaps.groupby("stage")["overlap_seconds"].sum()
            stage = stage_seconds.idxmax()

            row = {
                "night_date": night_date,
                "device": device,
                "bin_start": bin_start,
                "bin_end": bin_end,
                "stage": stage,
                "duration_minutes": bin_delta.total_seconds() / 60,
                "coverage_fraction": min(1.0, overlaps["overlap_seconds"].sum() / bin_delta.total_seconds()),
                "cpap_usage_fraction": (
                    overlaps.loc[overlaps["cpap_usage"], "overlap_seconds"].sum() / bin_delta.total_seconds()
                ),
            }
            for column in NUMERIC_COLUMNS:
                row[column] = _weighted_average(overlaps[column], overlaps["overlap_seconds"])
            for column in FLAG_COLUMNS:
                row[column] = bool(overlaps[column].any())
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["night_date", "bin_start", "device"]).reset_index(drop=True)
