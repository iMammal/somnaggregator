"""Quality and reporting helpers for extracted sleep observations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import (
    CANONICAL_METRICS,
    NIGHTLY_SUMMARY_COLUMNS,
    OBSERVATION_COLUMNS,
    PLOT_METRICS,
    SUMMARY_VALUE_METRICS,
    ensure_observations_frame,
    normalize_summary_columns,
)


CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def confidence_rank(label: object) -> int:
    """Return a sortable confidence rank."""

    return CONFIDENCE_RANK.get(str(label).lower(), 0)


def select_best_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep one best value per date/device/metric using confidence first."""

    df = ensure_observations_frame(observations)
    if df.empty:
        return df

    ranked = df.assign(_confidence_rank=df["confidence"].map(confidence_rank))
    ranked = ranked.sort_values(
        ["night_date", "device", "metric", "_confidence_rank", "extraction_method"],
        ascending=[True, True, True, False, True],
    )
    return ranked.drop_duplicates(["night_date", "device", "metric"], keep="first").drop(columns="_confidence_rank")


def observations_to_nightly_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form observations into one row per date/device."""

    best = select_best_observations(observations)
    if best.empty:
        return pd.DataFrame(columns=NIGHTLY_SUMMARY_COLUMNS)

    best = best.copy()
    best["night_date"] = best["night_date"].astype("object").where(best["night_date"].notna(), "undated")
    values = (
        best.pivot_table(index=["night_date", "device"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
    )
    for metric in SUMMARY_VALUE_METRICS:
        if metric not in values:
            values[metric] = pd.NA

    provenance = best.groupby(["night_date", "device"]).agg(
        source_files=("source_file", lambda values: "; ".join(sorted(set(map(str, values))))),
        extraction_methods=("extraction_method", lambda values: "; ".join(sorted(set(map(str, values))))),
        min_confidence=("confidence", _min_confidence),
        notes=("notes", _combine_notes),
    )
    summary = values.merge(provenance.reset_index(), on=["night_date", "device"], how="left")
    return normalize_summary_columns(summary).sort_values(["night_date", "device"]).reset_index(drop=True)


def missingness_by_device(nightly_summary: pd.DataFrame) -> pd.DataFrame:
    """Percent missing by device for each summary metric."""

    if nightly_summary.empty:
        return pd.DataFrame(columns=["device", "metric", "missing_pct"])

    rows = []
    for device, group in nightly_summary.groupby("device"):
        for metric in SUMMARY_VALUE_METRICS:
            rows.append(
                {
                    "device": device,
                    "metric": metric,
                    "missing_pct": float(group[metric].isna().mean() * 100) if metric in group else 100.0,
                }
            )
    return pd.DataFrame(rows)


def confidence_by_device(observations: pd.DataFrame) -> pd.DataFrame:
    """Count extracted values by device and confidence label."""

    df = ensure_observations_frame(observations)
    if df.empty:
        return pd.DataFrame(columns=["device", "confidence", "count"])
    return df.groupby(["device", "confidence"]).size().reset_index(name="count")


def describe_extraction_outputs(
    nightly_summary: pd.DataFrame,
    observations: pd.DataFrame,
    expected_plot_metrics: list[str] | None = None,
    print_output: bool = True,
) -> dict[str, object]:
    """Describe extracted outputs and clearly report plot-target availability."""

    expected = expected_plot_metrics or PLOT_METRICS
    summary = normalize_summary_columns(nightly_summary) if nightly_summary is not None else pd.DataFrame(columns=NIGHTLY_SUMMARY_COLUMNS)
    obs = ensure_observations_frame(observations) if observations is not None else pd.DataFrame(columns=OBSERVATION_COLUMNS)

    devices = sorted(set(summary.get("device", pd.Series(dtype=str)).dropna()) | set(obs.get("device", pd.Series(dtype=str)).dropna()))
    metric_names = sorted(obs["metric"].dropna().unique().tolist()) if not obs.empty else []
    canonical_available = [
        metric
        for metric in CANONICAL_METRICS
        if (metric in metric_names) or (metric in summary.columns and summary[metric].notna().any())
    ]
    expected_missing = [
        metric
        for metric in expected
        if metric not in summary.columns or not pd.to_numeric(summary[metric], errors="coerce").notna().any()
    ]
    source_files = sorted(obs["source_file"].dropna().unique().tolist()) if not obs.empty else []

    diagnostics = {
        "nightly_rows": int(len(summary)),
        "observation_rows": int(len(obs)),
        "devices_detected": devices,
        "metric_names_detected": metric_names,
        "canonical_metrics_available": canonical_available,
        "expected_plot_metrics_missing": expected_missing,
        "source_files": source_files,
        "nightly_summary_columns": list(summary.columns),
    }

    if print_output:
        print(f"Night/device rows: {diagnostics['nightly_rows']}")
        print(f"Observation rows: {diagnostics['observation_rows']}")
        print("Devices detected:", ", ".join(devices) if devices else "(none)")
        print("Metric names detected:", ", ".join(metric_names) if metric_names else "(none)")
        print("Canonical metrics available:", ", ".join(canonical_available) if canonical_available else "(none)")
        print("Expected plot metrics missing:", ", ".join(expected_missing) if expected_missing else "(none)")
        print("Source files contributing values:")
        if source_files:
            for source_file in source_files:
                print(f"  - {source_file}")
        else:
            print("  - (none)")

    return diagnostics


def write_extraction_outputs(
    observations: pd.DataFrame,
    processed_dir: Path,
    outputs_dir: Path,
    report_lines: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Write normalized CSVs and a markdown extraction report."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    long_df = ensure_observations_frame(observations)
    summary_df = observations_to_nightly_summary(long_df)

    summary_path = processed_dir / "nightly_summary.csv"
    long_path = processed_dir / "device_observations_long.csv"
    report_path = outputs_dir / "extraction_report.md"

    summary_df.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)
    report_path.write_text(build_extraction_report(summary_df, long_df, report_lines), encoding="utf-8")
    return summary_df, long_df, report_path


def build_extraction_report(
    nightly_summary: pd.DataFrame,
    observations: pd.DataFrame,
    report_lines: list[str] | None = None,
) -> str:
    """Create a compact markdown report for extraction transparency."""

    lines = [
        "# SleepPy Extraction Report",
        "",
        "This report summarizes automated first-pass extraction from screenshots and PDFs.",
        "",
        "This is exploratory wellness data analysis only. It is not medical diagnosis, treatment advice, or a replacement for clinician review.",
        "",
        f"- Night/device rows: {len(nightly_summary)}",
        f"- Extracted values: {len(observations)}",
    ]

    if not observations.empty:
        diagnostics = describe_extraction_outputs(nightly_summary, observations, print_output=False)
        lines.extend(["", "## Values By Device", ""])
        counts = observations.groupby("device").size().sort_index()
        lines.extend([f"- {device}: {count}" for device, count in counts.items()])

        lines.extend(["", "## Canonical Metrics Available", ""])
        if diagnostics["canonical_metrics_available"]:
            lines.extend([f"- {metric}" for metric in diagnostics["canonical_metrics_available"]])
        else:
            lines.append("- (none)")

        lines.extend(["", "## Confidence By Device", ""])
        confidence = confidence_by_device(observations)
        lines.extend(
            [
                f"- {row.device}: {row.confidence} = {row.count}"
                for row in confidence.itertuples(index=False)
            ]
        )

    if report_lines:
        lines.extend(["", "## File Notes", ""])
        lines.extend([f"- {line}" for line in report_lines])

    lines.extend(
        [
            "",
            "## OCR Setup",
            "",
            "PDF text is parsed with PyMuPDF when available. Image OCR uses pytesseract only when both the Python package and the Tesseract executable are installed.",
            "",
            "Install Python dependencies with:",
            "",
            "```powershell",
            "py -m pip install -r requirements.txt",
            "```",
            "",
            "On Windows, install the Tesseract executable separately and make sure `tesseract.exe` is on PATH, or set `pytesseract.pytesseract.tesseract_cmd` in a local notebook cell.",
        ]
    )
    return "\n".join(lines) + "\n"


def _min_confidence(values: pd.Series) -> str:
    labels = [str(value) for value in values.dropna()]
    if not labels:
        return "low"
    return min(labels, key=confidence_rank)


def _combine_notes(values: pd.Series) -> str:
    notes = [str(value).strip() for value in values.dropna() if str(value).strip()]
    return "; ".join(sorted(set(notes)))
