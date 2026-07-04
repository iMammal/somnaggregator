"""Comparison helpers for binned multi-device sleep data."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from .schema import NUMERIC_COLUMNS, STAGE_ORDER


def stage_pivot(binned: pd.DataFrame) -> pd.DataFrame:
    """Return a device-by-bin stage table for disagreement checks."""

    if binned.empty:
        return pd.DataFrame()
    return binned.pivot_table(
        index=["night_date", "bin_start"],
        columns="device",
        values="stage",
        aggfunc="first",
    )


def stage_disagreement_matrix(binned: pd.DataFrame) -> pd.DataFrame:
    """Pairwise percent disagreement for stage labels on shared bins."""

    stages = stage_pivot(binned)
    devices = list(stages.columns)
    matrix = pd.DataFrame(0.0, index=devices, columns=devices)

    for left, right in combinations(devices, 2):
        shared = stages[[left, right]].dropna()
        shared = shared[(shared[left] != "no_data") & (shared[right] != "no_data")]
        disagreement = 0.0 if shared.empty else float((shared[left] != shared[right]).mean() * 100)
        matrix.loc[left, right] = disagreement
        matrix.loc[right, left] = disagreement

    return matrix


def metric_pivot(binned: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one metric as bins by device."""

    if metric not in binned:
        raise ValueError(f"Metric {metric!r} is not present in the binned dataframe.")
    if binned.empty:
        return pd.DataFrame()
    return binned.pivot_table(
        index=["night_date", "bin_start"],
        columns="device",
        values=metric,
        aggfunc="mean",
    )


def summarize_by_device_night(binned: pd.DataFrame) -> pd.DataFrame:
    """Summarize stage minutes and available metrics by device and night."""

    if binned.empty:
        return pd.DataFrame()

    group_cols = ["night_date", "device"]
    stage_minutes = (
        binned.assign(minutes=binned["duration_minutes"] * binned["coverage_fraction"])
        .pivot_table(index=group_cols, columns="stage", values="minutes", aggfunc="sum", fill_value=0)
        .rename(columns={stage: f"{stage}_minutes" for stage in STAGE_ORDER})
    )

    metric_columns = [column for column in NUMERIC_COLUMNS if column in binned]
    metrics = binned.groupby(group_cols)[metric_columns].mean(numeric_only=True)
    flags = binned.groupby(group_cols).agg(
        cpap_usage_minutes=("cpap_usage_fraction", lambda values: float((values * binned.loc[values.index, "duration_minutes"]).sum())),
        event_bins=("event_flag", "sum"),
    )

    return pd.concat([stage_minutes, metrics, flags], axis=1).reset_index()
