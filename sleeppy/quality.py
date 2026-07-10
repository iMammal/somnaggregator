"""Quality and reporting helpers for extracted sleep observations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import (
    CANONICAL_METRICS,
    NIGHTLY_SUMMARY_COLUMNS,
    OBSERVATION_COLUMNS,
    OPTIONAL_DEVICE_METRICS,
    OPTIONAL_PLOT_METRICS,
    PLOT_METRICS,
    SUMMARY_VALUE_METRICS,
    ensure_observations_frame,
    normalize_summary_columns,
)


CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
MINDMONITOR_REPORT_PREFIX = "MINDMONITOR_REPORT:"


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
    return derive_duration_semantics(normalize_summary_columns(summary).sort_values(["night_date", "device"]).reset_index(drop=True))


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
    optional_plot_metrics = set(OPTIONAL_PLOT_METRICS)
    expected_missing = [
        metric
        for metric in expected
        if metric not in optional_plot_metrics
        and (metric not in summary.columns or not pd.to_numeric(summary[metric], errors="coerce").notna().any())
    ]
    optional_missing = [
        metric
        for metric in expected
        if metric in optional_plot_metrics
        and (metric not in summary.columns or not pd.to_numeric(summary[metric], errors="coerce").notna().any())
    ]
    cpap_metrics = OPTIONAL_DEVICE_METRICS["cpap"]
    cpap_metrics_detected = [
        metric
        for metric in cpap_metrics
        if (metric in metric_names) or (metric in summary.columns and summary[metric].notna().any())
    ]
    cpap_detected = bool(cpap_metrics_detected)
    source_files = sorted(obs["source_file"].dropna().unique().tolist()) if not obs.empty else []

    diagnostics = {
        "nightly_rows": int(len(summary)),
        "observation_rows": int(len(obs)),
        "devices_detected": devices,
        "metric_names_detected": metric_names,
        "canonical_metrics_available": canonical_available,
        "expected_plot_metrics_missing": expected_missing,
        "optional_plot_metrics_missing": optional_missing,
        "cpap_detected": cpap_detected,
        "cpap_metrics_detected": cpap_metrics_detected,
        "source_files": source_files,
        "nightly_summary_columns": list(summary.columns),
    }

    if print_output:
        print(f"Night/device rows: {diagnostics['nightly_rows']}")
        print(f"Observation rows: {diagnostics['observation_rows']}")
        print("Devices detected:", ", ".join(devices) if devices else "(none)")
        print("Metric names detected:", ", ".join(metric_names) if metric_names else "(none)")
        print("Canonical metrics available:", ", ".join(canonical_available) if canonical_available else "(none)")
        if cpap_detected:
            print("CPAP metrics detected:", ", ".join(cpap_metrics_detected))
        else:
            print("No CPAP metrics detected; CPAP/OSCAR/SleepScope is optional.")
        print("Required plot metrics missing:", ", ".join(expected_missing) if expected_missing else "(none)")
        if optional_missing:
            print("Optional plot metrics unavailable:", ", ".join(optional_missing))
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
        "This report summarizes automated first-pass extraction from screenshots, PDFs, and CSV sensor logs.",
        "",
        "This is exploratory wellness data analysis only. It is not medical diagnosis, treatment advice, or a replacement for clinician review.",
        "",
        f"- Night/device rows: {len(nightly_summary)}",
        f"- Extracted values: {len(observations)}",
    ]

    diagnostics = describe_extraction_outputs(nightly_summary, observations, print_output=False)
    if not observations.empty:
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

    lines.extend(["", "## Optional CPAP Status", ""])
    if diagnostics["cpap_detected"]:
        lines.extend([f"- CPAP metrics detected: {', '.join(diagnostics['cpap_metrics_detected'])}"])
    else:
        lines.append("- No CPAP metrics detected; CPAP/OSCAR/SleepScope is optional.")

    mindmonitor_report = _mindmonitor_report_from_lines(report_lines) or _mindmonitor_report_from_observations(observations)
    if mindmonitor_report is not None:
        lines.extend(["", "## MindMonitor", ""])
        lines.extend(_format_mindmonitor_report(mindmonitor_report))

    if report_lines:
        visible_report_lines = [line for line in report_lines if not str(line).startswith(MINDMONITOR_REPORT_PREFIX)]
        lines.extend(["", "## File Notes", ""])
        lines.extend([f"- {line}" for line in visible_report_lines])

    warnings = check_physiological_sanity(nightly_summary)
    if warnings:
        lines.extend(["", "## Extraction Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])

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


def _mindmonitor_report_from_lines(report_lines: list[str] | None) -> dict[str, object] | None:
    if not report_lines:
        return None
    for line in report_lines:
        text = str(line)
        if not text.startswith(MINDMONITOR_REPORT_PREFIX):
            continue
        try:
            report = json.loads(text.removeprefix(MINDMONITOR_REPORT_PREFIX))
        except json.JSONDecodeError:
            return None
        return report if isinstance(report, dict) else None
    return None


def _mindmonitor_report_from_observations(observations: pd.DataFrame) -> dict[str, object] | None:
    if observations.empty or "device" not in observations:
        return None
    obs = observations[observations["device"].astype(str).eq("Muse S MindMonitor")]
    if obs.empty:
        return None
    rows_parsed = _sum_metric(obs, "mindmonitor_rows")
    session_minutes = _sum_metric(obs, "mindmonitor_session_minutes")
    return {
        "files_detected": int(obs["source_file"].nunique()),
        "rows_parsed": int(rows_parsed) if rows_parsed is not None else 0,
        "observations_extracted": int(len(obs)),
        "channel_groups": _infer_mindmonitor_channel_groups(obs),
        "sessions": [
            {
                "source_file": str(source_file),
                "rows_parsed": int(_sum_metric(group, "mindmonitor_rows") or 0),
                "observations_extracted": int(len(group)),
                "session_start": _first_metric(group, "mindmonitor_session_start_time"),
                "session_end": _first_metric(group, "mindmonitor_session_end_time"),
                "session_minutes": _sum_metric(group, "mindmonitor_session_minutes"),
                "crossed_midnight": _session_crossed_midnight_from_group(group),
                "stopped_before_morning": bool(_sum_metric(group, "mindmonitor_stopped_before_morning") or 0),
                "gap_count_gt_5s": int(_sum_metric(group, "mindmonitor_gap_count_gt_5s") or 0),
                "max_gap_seconds": _sum_metric(group, "mindmonitor_max_gap_seconds"),
                "battery_min": _sum_metric(group, "mindmonitor_battery_min"),
                "battery_max": _sum_metric(group, "mindmonitor_battery_max"),
                "valid_eeg_rows": int(_sum_metric(group, "mindmonitor_valid_eeg_rows") or 0),
                "valid_motion_rows": int(_sum_metric(group, "mindmonitor_valid_motion_rows") or 0),
                "valid_ppg_rows": int(_sum_metric(group, "mindmonitor_valid_ppg_rows") or 0),
                "error": None,
            }
            for source_file, group in obs.groupby("source_file", sort=True)
        ],
        "session_minutes": session_minutes,
    }


def _format_mindmonitor_report(report: dict[str, object]) -> list[str]:
    files_detected = int(report.get("files_detected") or 0)
    rows_parsed = int(report.get("rows_parsed") or 0)
    observations_extracted = int(report.get("observations_extracted") or 0)
    channel_groups = report.get("channel_groups") or []
    if not isinstance(channel_groups, list):
        channel_groups = []

    lines = [
        f"- Files detected: {files_detected}",
        f"- Rows parsed: {rows_parsed}",
        f"- Channel groups detected: {', '.join(map(str, channel_groups)) if channel_groups else '(none)'}",
        f"- Observations extracted: {observations_extracted}",
    ]

    sessions = report.get("sessions") or []
    if isinstance(sessions, list) and sessions:
        lines.append("- Session duration:")
        for session in sessions:
            if not isinstance(session, dict):
                continue
            source = session.get("source_file") or "(unknown source)"
            error = session.get("error")
            if error:
                lines.append(f"  - {source}: parse failed ({error})")
                continue
            duration = session.get("session_minutes")
            start = session.get("session_start") or "unknown start"
            end = session.get("session_end") or "unknown end"
            lines.append(f"  - {source}: {duration if duration is not None else 'unknown'} minutes ({start} to {end})")
            lines.append(f"    - Crossed midnight: {_yes_no(session.get('crossed_midnight'))}")
            lines.append(
                f"    - Truncated before morning (<04:00 end): {_yes_no(session.get('stopped_before_morning'))}"
            )
            lines.append(
                f"    - Battery min/max: {_format_optional_number(session.get('battery_min'))} / "
                f"{_format_optional_number(session.get('battery_max'))}"
            )
            lines.append(
                f"    - Valid rows: EEG={session.get('valid_eeg_rows', 0)}, "
                f"motion={session.get('valid_motion_rows', 0)}, PPG={session.get('valid_ppg_rows', 0)}"
            )
            lines.append(
                f"    - Gaps >5s: {session.get('gap_count_gt_5s', 0)}; "
                f"max gap seconds: {_format_optional_number(session.get('max_gap_seconds'))}"
            )
    elif files_detected == 0:
        lines.append("- Session duration: no MindMonitor CSV files detected.")
    else:
        lines.append("- Session duration: unavailable.")
    return lines


def _sum_metric(observations: pd.DataFrame, metric: str) -> float | None:
    if observations.empty:
        return None
    values = pd.to_numeric(observations.loc[observations["metric"].eq(metric), "value"], errors="coerce")
    if values.dropna().empty:
        return None
    return float(values.sum())


def _first_metric(observations: pd.DataFrame, metric: str) -> object | None:
    values = observations.loc[observations["metric"].eq(metric), "value"]
    values = values.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _session_crossed_midnight_from_group(observations: pd.DataFrame) -> bool:
    start = pd.to_datetime(_first_metric(observations, "mindmonitor_session_start_time"), errors="coerce")
    end = pd.to_datetime(_first_metric(observations, "mindmonitor_session_end_time"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return False
    return start.date() != end.date()


def _yes_no(value: object) -> str:
    if isinstance(value, str):
        return "yes" if value.strip().lower() in {"true", "1", "yes"} else "no"
    return "yes" if bool(value) else "no"


def _format_optional_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _infer_mindmonitor_channel_groups(observations: pd.DataFrame) -> list[str]:
    metrics = set(observations["metric"].astype(str))
    groups = []
    if "mindmonitor_valid_eeg_rows" in metrics:
        groups.append("eeg_raw")
    if any(metric in metrics for metric in ["mindmonitor_mean_delta", "mindmonitor_mean_theta", "mindmonitor_mean_alpha", "mindmonitor_mean_beta", "mindmonitor_mean_gamma"]):
        groups.append("bandpower")
    if any(metric in metrics for metric in ["mindmonitor_mean_accel_mag", "mindmonitor_p95_accel_mag"]):
        groups.append("accelerometer")
    if any(metric in metrics for metric in ["mindmonitor_mean_gyro_mag", "mindmonitor_p95_gyro_mag"]):
        groups.append("gyroscope")
    if "mindmonitor_valid_ppg_rows" in metrics:
        groups.append("ppg")
    if any(metric in metrics for metric in ["mindmonitor_mean_hr_bpm", "mindmonitor_median_hr_bpm"]):
        groups.append("heart_rate")
    if any(metric.startswith("mindmonitor_mean_hsi_") for metric in metrics) or "mindmonitor_headband_on_fraction" in metrics:
        groups.append("hsi/contact")
    if any(metric in metrics for metric in ["mindmonitor_battery_min", "mindmonitor_battery_max"]):
        groups.append("battery")
    return groups


def _min_confidence(values: pd.Series) -> str:
    labels = [str(value) for value in values.dropna()]
    if not labels:
        return "low"
    return min(labels, key=confidence_rank)


def _combine_notes(values: pd.Series) -> str:
    notes = [str(value).strip() for value in values.dropna() if str(value).strip()]
    return "; ".join(sorted(set(notes)))


def _append_note(existing: object, note: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing).strip()
    if not text:
        return note
    if note in text:
        return text
    return f"{text}; {note}"


def _numeric_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _store_numeric(row: pd.Series, key: str, value: float, note: str | None = None) -> None:
    if float(value).is_integer():
        row[key] = int(value)
    else:
        row[key] = round(value, 3)
    if note:
        row["notes"] = _append_note(row.get("notes"), note)


def _normalize_oura_duration_row(row: pd.Series) -> pd.Series:
    """Reconcile Oura duration metrics against the strongest available evidence."""

    row = row.copy()
    total_sleep = _numeric_value(row.get("total_sleep_minutes"))
    time_in_bed = _numeric_value(row.get("time_in_bed_minutes"))
    efficiency = _numeric_value(row.get("sleep_efficiency_pct"))
    awake = _numeric_value(row.get("awake_minutes"))
    rem = _numeric_value(row.get("rem_minutes"))
    deep = _numeric_value(row.get("deep_minutes"))
    light = _numeric_value(row.get("light_minutes"))

    if total_sleep is not None and efficiency is not None and efficiency > 0:
        candidate_tib = round(total_sleep * 100.0 / efficiency)
        if time_in_bed is None or abs(time_in_bed - candidate_tib) > 10:
            _store_numeric(row, "time_in_bed_minutes", candidate_tib, "time_in_bed_minutes corrected from Oura efficiency")
            time_in_bed = float(candidate_tib)

    if time_in_bed is None and total_sleep is not None and awake is not None:
        candidate_tib = round(total_sleep + awake)
        _store_numeric(row, "time_in_bed_minutes", candidate_tib, "time_in_bed_minutes derived from total_sleep+awake")
        time_in_bed = float(candidate_tib)

    if time_in_bed is not None and awake is not None:
        candidate_sleep = round(time_in_bed - awake)
        if candidate_sleep > 0 and (total_sleep is None or abs(total_sleep - candidate_sleep) > 10):
            _store_numeric(row, "total_sleep_minutes", candidate_sleep, "total_sleep_minutes corrected from time_in_bed-awake")
            total_sleep = float(candidate_sleep)

    if time_in_bed is not None and total_sleep is not None and awake is None:
        candidate_awake = round(time_in_bed - total_sleep)
        if candidate_awake >= 0:
            _store_numeric(row, "awake_minutes", candidate_awake, "awake_minutes derived from time_in_bed-total_sleep")
            awake = float(candidate_awake)

    if time_in_bed is not None:
        candidate_light = None
        if awake is not None and rem is not None and deep is not None:
            candidate_light = round(time_in_bed - awake - rem - deep)
        elif total_sleep is not None and rem is not None and deep is not None:
            candidate_light = round(total_sleep - rem - deep)
        if candidate_light is not None and candidate_light >= 0:
            if light is None or abs(light - candidate_light) > 10:
                _store_numeric(row, "light_minutes", candidate_light, "light_minutes corrected from Oura duration consistency")
                light = float(candidate_light)

    total_sleep = _numeric_value(row.get("total_sleep_minutes"))
    time_in_bed = _numeric_value(row.get("time_in_bed_minutes"))
    if time_in_bed is not None and total_sleep is not None and time_in_bed > 0:
        candidate_efficiency = round(total_sleep * 100.0 / time_in_bed)
        current_efficiency = _numeric_value(row.get("sleep_efficiency_pct"))
        if current_efficiency is None or abs(current_efficiency - candidate_efficiency) > 2:
            _store_numeric(row, "sleep_efficiency_pct", candidate_efficiency, "sleep_efficiency_pct derived from total_sleep/time_in_bed")

    return row


def derive_duration_semantics(summary: pd.DataFrame) -> pd.DataFrame:
    """Derive time_in_bed_minutes from total_sleep_minutes and sleep_efficiency_pct if needed."""

    summary = summary.copy()
    if "time_in_bed_minutes" not in summary.columns:
        summary["time_in_bed_minutes"] = pd.NA
    summary["time_in_bed_minutes"] = pd.to_numeric(summary["time_in_bed_minutes"], errors="coerce")
    
    if "notes" not in summary.columns:
        summary["notes"] = pd.NA
    summary["notes"] = summary["notes"].astype("string")

    # 1. Derive from efficiency if missing
    mask = (summary["device"].str.startswith("Oura", na=False)) & \
           (summary["time_in_bed_minutes"].isna()) & \
           (summary["total_sleep_minutes"].notna()) & \
           (summary["sleep_efficiency_pct"].notna())

    if mask.any():
        summary.loc[mask, "time_in_bed_minutes"] = (
            summary.loc[mask, "total_sleep_minutes"] / (summary.loc[mask, "sleep_efficiency_pct"] / 100.0)
        ).round()
        summary.loc[mask, "notes"] = summary.loc[mask, "notes"].fillna("") + "; time_in_bed_minutes derived from efficiency"

    # 2. Correct if total_sleep + awake implies a larger time_in_bed
    if "awake_minutes" in summary.columns:
        mask_awake = (summary["device"].str.startswith("Oura", na=False)) & \
                     (summary["total_sleep_minutes"].notna()) & \
                     (summary["awake_minutes"].notna())

        # Calculate correction mask on full summary to avoid index alignment issues
        total_sleep = pd.to_numeric(summary["total_sleep_minutes"], errors="coerce")
        awake = pd.to_numeric(summary["awake_minutes"], errors="coerce")
        current_tib = pd.to_numeric(summary["time_in_bed_minutes"], errors="coerce")
        
        derived_tib = total_sleep + awake
        mask_correction = current_tib.isna() | (derived_tib > current_tib) | ((derived_tib - current_tib).abs() <= 1)
        
        mask_final = mask_awake & mask_correction
        if mask_final.any():
            summary.loc[mask_final, "time_in_bed_minutes"] = derived_tib[mask_final]
            summary.loc[mask_final, "notes"] = (
                summary.loc[mask_final, "notes"].fillna("") + "; time_in_bed_minutes corrected from total_sleep+awake"
            )

    if "device" in summary.columns:
        oura_mask = summary["device"].astype(str).str.startswith("Oura", na=False)
        if oura_mask.any():
            # Apply only to relevant rows, and assign column by column to avoid dtype mismatch
            updated = summary.loc[oura_mask].apply(_normalize_oura_duration_row, axis=1)
            for col in updated.columns:
                if col in summary.columns:
                    target_dtype = summary[col].dtype
                    summary.loc[oura_mask, col] = updated[col].astype(target_dtype)

    return summary

def check_physiological_sanity(nightly_summary: pd.DataFrame) -> list[str]:
    warnings = []
    if nightly_summary.empty:
        return warnings

    nightly_summary = nightly_summary.copy()
    numeric_columns = [
        "avg_spo2_pct",
        "avg_hr_bpm",
        "min_hr_bpm",
        "sleep_efficiency_pct",
        "total_sleep_minutes",
        "time_in_bed_minutes",
        "rem_minutes",
        "light_minutes",
        "deep_minutes",
        "awake_minutes",
    ]
    for column in numeric_columns:
        if column in nightly_summary.columns:
            nightly_summary[column] = pd.to_numeric(nightly_summary[column], errors="coerce")

    # - avg_spo2_pct should usually be 70-100
    if "avg_spo2_pct" in nightly_summary.columns:
        suspicious = nightly_summary[~nightly_summary["avg_spo2_pct"].isna() &
                                     ((nightly_summary["avg_spo2_pct"] < 70) | (nightly_summary["avg_spo2_pct"] > 100))]
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious avg_spo2_pct={row['avg_spo2_pct']} on {row['night_date']} for {row['device']}. Check OCR/source image.")

    # - avg_hr_bpm should usually be 30-220
    if "avg_hr_bpm" in nightly_summary.columns:
        suspicious = nightly_summary[~nightly_summary["avg_hr_bpm"].isna() &
                                     ((nightly_summary["avg_hr_bpm"] < 30) | (nightly_summary["avg_hr_bpm"] > 220))]
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious avg_hr_bpm={row['avg_hr_bpm']} on {row['night_date']} for {row['device']}.")

    # - min_hr_bpm should usually be 25-180
    if "min_hr_bpm" in nightly_summary.columns:
        suspicious = nightly_summary[~nightly_summary["min_hr_bpm"].isna() &
                                     ((nightly_summary["min_hr_bpm"] < 25) | (nightly_summary["min_hr_bpm"] > 180))]
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious min_hr_bpm={row['min_hr_bpm']} on {row['night_date']} for {row['device']}.")

    # - sleep_efficiency_pct should be 0-100
    if "sleep_efficiency_pct" in nightly_summary.columns:
        suspicious = nightly_summary[~nightly_summary["sleep_efficiency_pct"].isna() &
                                     ((nightly_summary["sleep_efficiency_pct"] < 0) | (nightly_summary["sleep_efficiency_pct"] > 100))]
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious sleep_efficiency_pct={row['sleep_efficiency_pct']} on {row['night_date']} for {row['device']}.")

    # - total_sleep_minutes should be 0-1440
    if "total_sleep_minutes" in nightly_summary.columns:
        suspicious = nightly_summary[~nightly_summary["total_sleep_minutes"].isna() &
                                     ((nightly_summary["total_sleep_minutes"] < 0) | (nightly_summary["total_sleep_minutes"] > 1440))]
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious total_sleep_minutes={row['total_sleep_minutes']} on {row['night_date']} for {row['device']}.")

    # - time_in_bed_minutes should be >= total_sleep_minutes when both are available
    if "time_in_bed_minutes" in nightly_summary.columns and "total_sleep_minutes" in nightly_summary.columns:
        suspicious = nightly_summary[~nightly_summary["time_in_bed_minutes"].isna() & ~nightly_summary["total_sleep_minutes"].isna() &
                                     (nightly_summary["time_in_bed_minutes"] < nightly_summary["total_sleep_minutes"])]
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious: time_in_bed_minutes={row['time_in_bed_minutes']} < total_sleep_minutes={row['total_sleep_minutes']} on {row['night_date']} for {row['device']}.")

    # - REM + light + deep should approximately equal total_sleep_minutes when sleep stages are available
    stage_cols = {"rem_minutes", "light_minutes", "deep_minutes"}
    if stage_cols.issubset(nightly_summary.columns) and "total_sleep_minutes" in nightly_summary.columns:
        df = nightly_summary.dropna(subset=[*stage_cols, "total_sleep_minutes"])
        df = df.assign(sum_stages=df["rem_minutes"] + df["light_minutes"] + df["deep_minutes"])
        suspicious = df[abs(df["sum_stages"] - df["total_sleep_minutes"]) > 15] # Allowing 15 mins discrepancy
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious: stage sum={row['sum_stages']} != total_sleep_minutes={row['total_sleep_minutes']} on {row['night_date']} for {row['device']}.")

    # - awake + REM + light + deep should approximately equal time_in_bed_minutes
    stage_cols_bed = {"rem_minutes", "light_minutes", "deep_minutes", "awake_minutes"}
    if stage_cols_bed.issubset(nightly_summary.columns) and "time_in_bed_minutes" in nightly_summary.columns:
        df = nightly_summary.dropna(subset=[*stage_cols_bed, "time_in_bed_minutes"])
        df = df.assign(sum_stages_bed=df["rem_minutes"] + df["light_minutes"] + df["deep_minutes"] + df["awake_minutes"])
        suspicious = df[abs(df["sum_stages_bed"] - df["time_in_bed_minutes"]) > 15] # Allowing 15 mins discrepancy
        for _, row in suspicious.iterrows():
            warnings.append(f"Suspicious: stage sum={row['sum_stages_bed']} != time_in_bed_minutes={row['time_in_bed_minutes']} on {row['night_date']} for {row['device']}.")

    # - Oura specific: total_sleep_minutes + awake_minutes should approximately equal time_in_bed_minutes
    # - Oura specific: sleep_efficiency_pct should approximately equal total_sleep_minutes / time_in_bed_minutes * 100
    oura_mask = nightly_summary["device"].str.startswith("Oura", na=False)
    oura_summary = nightly_summary[oura_mask].dropna(subset=["total_sleep_minutes", "time_in_bed_minutes"])
    for _, row in oura_summary.iterrows():
        # Check: total_sleep + awake approx time_in_bed
        if "awake_minutes" in row and pd.notna(row["awake_minutes"]):
            if abs((row["total_sleep_minutes"] + row["awake_minutes"]) - row["time_in_bed_minutes"]) > 5:
                warnings.append(f"Suspicious Oura duration: total_sleep={row['total_sleep_minutes']} + awake={row['awake_minutes']} != time_in_bed={row['time_in_bed_minutes']} on {row['night_date']}.")
        
        # Check: efficiency approx total_sleep / time_in_bed * 100
        if "sleep_efficiency_pct" in row and pd.notna(row["sleep_efficiency_pct"]) and row["time_in_bed_minutes"] > 0:
            expected_eff = (row["total_sleep_minutes"] / row["time_in_bed_minutes"]) * 100
            if abs(row["sleep_efficiency_pct"] - expected_eff) > 5:
                warnings.append(f"Suspicious Oura efficiency: {row['sleep_efficiency_pct']}% != {row['total_sleep_minutes']}/{row['time_in_bed_minutes']}={expected_eff:.1f}% on {row['night_date']}.")

    return warnings
