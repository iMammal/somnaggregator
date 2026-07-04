"""Matplotlib visualizations for sleep timelines and device comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from .compare import stage_disagreement_matrix
from .schema import STAGE_COLORS, STAGE_LABELS, STAGE_ORDER


def _filter_binned(
    binned: pd.DataFrame,
    night_date: str | None = None,
    devices: list[str] | None = None,
) -> pd.DataFrame:
    df = binned.copy()
    if night_date is not None and not df.empty:
        df = df[pd.to_datetime(df["night_date"]) == pd.to_datetime(night_date)]
    if devices is not None and not df.empty:
        df = df[df["device"].isin(devices)]
    return df


def plot_sleep_stage_timeline(
    binned: pd.DataFrame,
    night_date: str | None = None,
    devices: list[str] | None = None,
    ax=None,
):
    """Plot device sleep stages as one row per device over clock time."""

    df = _filter_binned(binned, night_date=night_date, devices=devices)
    if df.empty:
        raise ValueError("No binned data available for the requested filters.")

    table = df.pivot_table(index="device", columns="bin_start", values="stage", aggfunc="first").fillna("no_data")
    stage_to_code = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    matrix = table.replace(stage_to_code).astype(float).to_numpy()
    cmap = ListedColormap([STAGE_COLORS[stage] for stage in STAGE_ORDER])

    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(2.5, 0.45 * len(table.index))))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(STAGE_ORDER) - 1)
    ax.set_yticks(np.arange(len(table.index)), table.index)
    ax.set_ylabel("Device")
    ax.set_title("Sleep-stage timeline")

    tick_positions = np.linspace(0, len(table.columns) - 1, min(8, len(table.columns)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pd.Timestamp(table.columns[position]).strftime("%H:%M") for position in tick_positions], rotation=45)
    ax.set_xlabel("Clock time")

    colorbar = ax.figure.colorbar(image, ax=ax, ticks=np.arange(len(STAGE_ORDER)))
    colorbar.ax.set_yticklabels([STAGE_LABELS[stage] for stage in STAGE_ORDER])
    ax.figure.tight_layout()
    return ax.figure, ax


def plot_hr_hrv_overlay(
    binned: pd.DataFrame,
    night_date: str | None = None,
    devices: list[str] | None = None,
    ax=None,
):
    """Overlay heart rate and HRV by device."""

    df = _filter_binned(binned, night_date=night_date, devices=devices)
    if df.empty:
        raise ValueError("No binned data available for the requested filters.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    hrv_ax = ax.twinx()

    for device, group in df.groupby("device", sort=True):
        group = group.sort_values("bin_start")
        if group["hr"].notna().any():
            ax.plot(group["bin_start"], group["hr"], marker="o", markersize=3, label=f"{device} HR")
        if group["hrv"].notna().any():
            hrv_ax.plot(group["bin_start"], group["hrv"], linestyle="--", marker=".", label=f"{device} HRV")

    ax.set_ylabel("Heart rate")
    hrv_ax.set_ylabel("HRV")
    ax.set_xlabel("Clock time")
    ax.set_title("HR and HRV overlay")
    ax.tick_params(axis="x", rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    hrv_handles, hrv_labels = hrv_ax.get_legend_handles_labels()
    if handles or hrv_handles:
        ax.legend(handles + hrv_handles, labels + hrv_labels, loc="best")
    ax.figure.tight_layout()
    return ax.figure, (ax, hrv_ax)


def plot_cpap_panel(
    binned: pd.DataFrame,
    night_date: str | None = None,
    devices: list[str] | None = None,
):
    """Plot CPAP usage, pressure/leak, and event markers."""

    df = _filter_binned(binned, night_date=night_date, devices=devices)
    if df.empty:
        raise ValueError("No binned data available for the requested filters.")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for device, group in df.groupby("device", sort=True):
        group = group.sort_values("bin_start")
        if group["cpap_usage_fraction"].notna().any():
            axes[0].step(group["bin_start"], group["cpap_usage_fraction"], where="post", label=device)
        if group["pressure"].notna().any():
            axes[1].plot(group["bin_start"], group["pressure"], marker="o", markersize=3, label=f"{device} pressure")
        if group["leak"].notna().any():
            axes[1].plot(group["bin_start"], group["leak"], linestyle="--", marker=".", label=f"{device} leak")

        event_rows = group[group["event_flag"] | (group["event_count"].fillna(0) > 0)]
        if not event_rows.empty:
            axes[2].scatter(event_rows["bin_start"], np.repeat(device, len(event_rows)), label=device)

    axes[0].set_ylabel("CPAP use")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Pressure / leak")
    axes[2].set_ylabel("Events")
    axes[2].set_xlabel("Clock time")
    axes[0].set_title("CPAP usage, pressure, leak, and event markers")
    for axis in axes:
        axis.legend(loc="best")
        axis.grid(True, alpha=0.25)
    axes[-1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, axes


def plot_stage_disagreement_heatmap(binned: pd.DataFrame, ax=None):
    """Plot pairwise percent disagreement between device stage labels."""

    matrix = stage_disagreement_matrix(binned)
    if matrix.empty:
        raise ValueError("Need at least one device with binned stage data.")

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, len(matrix) * 1.2), max(3, len(matrix) * 0.9)))
    image = ax.imshow(matrix.to_numpy(), cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Stage disagreement (%)")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix.iat[row, column]:.0f}", ha="center", va="center", color="white")

    ax.figure.colorbar(image, ax=ax, label="% disagreement")
    ax.figure.tight_layout()
    return ax.figure, ax
