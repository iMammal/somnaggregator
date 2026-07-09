"""MindMonitor CSV sensor-stream extraction."""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .common import observation, source_file_label


DEVICE_NAME = "Muse S MindMonitor"
SUPPORTED_EXTENSIONS = {".csv", ".zip"}

FILENAME_DATE_RE = re.compile(
    r"museMonitor_(?P<date>20\d{2}-\d{2}-\d{2})--(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})",
    flags=re.IGNORECASE,
)

EEG_RAW_COLUMNS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
ELECTRODES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
ACCEL_COLUMNS = ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"]
GYRO_COLUMNS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
PPG_COLUMNS = ["PPG_Ambient", "PPG_IR", "PPG_Red"]
HSI_COLUMNS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]


@dataclass
class MindMonitorDiagnostics:
    """Parser diagnostics used by the extraction report."""

    source_file: str
    files_detected: int
    rows_parsed: int
    observations_extracted: int
    session_start: str | None
    session_end: str | None
    session_minutes: float | None
    channel_groups: list[str]
    columns_present: list[str]
    valid_eeg_rows: int
    valid_motion_rows: int
    valid_ppg_rows: int
    notes: str
    error: str | None = None

    def to_report_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "files_detected": self.files_detected,
            "rows_parsed": self.rows_parsed,
            "observations_extracted": self.observations_extracted,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "session_minutes": self.session_minutes,
            "channel_groups": self.channel_groups,
            "columns_present": self.columns_present,
            "valid_eeg_rows": self.valid_eeg_rows,
            "valid_motion_rows": self.valid_motion_rows,
            "valid_ppg_rows": self.valid_ppg_rows,
            "notes": self.notes,
            "error": self.error,
        }


def find_files(folder: str | Path) -> list[Path]:
    """Find MindMonitor CSV files and single-CSV ZIP archives recursively."""

    root = Path(folder)
    if not root.exists():
        return []
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    csv_stems = {
        path.with_suffix("").resolve()
        for path in candidates
        if path.suffix.lower() == ".csv"
    }
    return [
        path
        for path in candidates
        if path.suffix.lower() != ".zip" or path.with_suffix("").resolve() not in csv_stems
    ]


def extract_file(path: str | Path) -> list[dict[str, object]]:
    """Extract long-form observations from one MindMonitor CSV or ZIP file."""

    rows, _diagnostics = extract_file_with_details(path)
    return rows


def extract_file_with_details(path: str | Path) -> tuple[list[dict[str, object]], MindMonitorDiagnostics]:
    """Extract observations plus report diagnostics from one MindMonitor file."""

    source_path = Path(path)
    source_label = source_file_label(source_path)
    try:
        df, source_label = _read_mindmonitor_table(source_path, source_label)
    except Exception as exc:
        diagnostics = MindMonitorDiagnostics(
            source_file=source_label,
            files_detected=1,
            rows_parsed=0,
            observations_extracted=0,
            session_start=None,
            session_end=None,
            session_minutes=None,
            channel_groups=[],
            columns_present=[],
            valid_eeg_rows=0,
            valid_motion_rows=0,
            valid_ppg_rows=0,
            notes=f"MindMonitor CSV parse failed; no sleep staging performed. Error: {exc}",
            error=str(exc),
        )
        return [], diagnostics

    rows, diagnostics = parse_mindmonitor_frame(df, source_file=source_label)
    return rows, diagnostics


def parse_mindmonitor_frame(
    frame: pd.DataFrame,
    *,
    source_file: str,
) -> tuple[list[dict[str, object]], MindMonitorDiagnostics]:
    """Parse a MindMonitor dataframe into conservative session-level observations."""

    df = frame.copy()
    df.columns = [_clean_column_name(column) for column in df.columns]
    columns_present = list(df.columns)
    timestamps = _parse_timestamps(df)
    session_start = timestamps.min() if timestamps.notna().any() else pd.NaT
    session_end = timestamps.max() if timestamps.notna().any() else pd.NaT
    session_minutes = _session_minutes(session_start, session_end)
    night_date = infer_night_date(source_file, timestamps)

    valid_eeg_rows = _valid_any_numeric_rows(df, EEG_RAW_COLUMNS)
    valid_ppg_rows = _valid_any_numeric_rows(df, PPG_COLUMNS)
    valid_motion_rows = int((_complete_numeric_rows(df, ACCEL_COLUMNS) | _complete_numeric_rows(df, GYRO_COLUMNS)).sum())
    channel_groups = detected_channel_groups(df)

    notes = _build_notes(
        session_start=session_start,
        session_end=session_end,
        columns_present=columns_present,
        valid_eeg_rows=valid_eeg_rows,
        valid_motion_rows=valid_motion_rows,
        valid_ppg_rows=valid_ppg_rows,
    )

    rows: list[dict[str, object]] = []

    def add_metric(metric: str, value: object, unit: str) -> None:
        if value is None:
            return
        rows.append(
            observation(
                date=night_date,
                device=DEVICE_NAME,
                metric=metric,
                value=_store_value(value),
                unit=unit,
                source_file=source_file,
                extraction_method="csv",
                confidence="medium",
                notes=notes,
            )
        )

    add_metric("mindmonitor_session_minutes", session_minutes, "minutes")
    add_metric("mindmonitor_rows", len(df), "rows")
    add_metric("mindmonitor_valid_eeg_rows", valid_eeg_rows, "rows")
    add_metric("mindmonitor_valid_motion_rows", valid_motion_rows, "rows")
    add_metric("mindmonitor_valid_ppg_rows", valid_ppg_rows, "rows")

    heart_rate = _numeric_series(df, "Heart_Rate")
    heart_rate = heart_rate[(heart_rate > 0) & (heart_rate >= 20) & (heart_rate <= 250)]
    add_metric("mindmonitor_mean_hr_bpm", _mean(heart_rate), "bpm")
    add_metric("mindmonitor_median_hr_bpm", _median(heart_rate), "bpm")

    accel_mag = _magnitude(df, ACCEL_COLUMNS)
    add_metric("mindmonitor_mean_accel_mag", _mean(accel_mag), "magnitude")
    add_metric("mindmonitor_p95_accel_mag", _quantile(accel_mag, 0.95), "magnitude")

    gyro_mag = _magnitude(df, GYRO_COLUMNS)
    add_metric("mindmonitor_mean_gyro_mag", _mean(gyro_mag), "magnitude")
    add_metric("mindmonitor_p95_gyro_mag", _quantile(gyro_mag, 0.95), "magnitude")

    for column in HSI_COLUMNS:
        add_metric(f"mindmonitor_mean_hsi_{column.removeprefix('HSI_').lower()}", _mean(_numeric_series(df, column)), "quality")

    add_metric("mindmonitor_headband_on_fraction", _headband_on_fraction(df), "fraction")
    add_metric("mindmonitor_battery_min", _min(_numeric_series(df, "Battery")), "pct")
    add_metric("mindmonitor_battery_max", _max(_numeric_series(df, "Battery")), "pct")

    for band in BANDS:
        band_mean = _band_mean(df, band)
        if band_mean is not None:
            add_metric(f"mindmonitor_mean_{band.lower()}", band_mean, "value")

    diagnostics = MindMonitorDiagnostics(
        source_file=source_file,
        files_detected=1,
        rows_parsed=int(len(df)),
        observations_extracted=len(rows),
        session_start=_format_timestamp(session_start),
        session_end=_format_timestamp(session_end),
        session_minutes=session_minutes,
        channel_groups=channel_groups,
        columns_present=columns_present,
        valid_eeg_rows=valid_eeg_rows,
        valid_motion_rows=valid_motion_rows,
        valid_ppg_rows=valid_ppg_rows,
        notes=notes,
    )
    return rows, diagnostics


def infer_night_date(source_file: str | Path, timestamps: pd.Series | None = None) -> str | None:
    """Infer night_date from MindMonitor filename first, then first valid TimeStamp."""

    match = FILENAME_DATE_RE.search(str(source_file))
    if match:
        return match.group("date")
    if timestamps is not None and timestamps.notna().any():
        return timestamps.dropna().iloc[0].date().isoformat()
    return None


def detected_channel_groups(frame: pd.DataFrame) -> list[str]:
    """Return channel groups represented by present columns."""

    columns = set(frame.columns)
    groups: list[str] = []
    if any(column in columns for column in EEG_RAW_COLUMNS):
        groups.append("eeg_raw")
    if any(f"{band}_{electrode}" in columns for band in BANDS for electrode in ELECTRODES):
        groups.append("bandpower")
    if any(column in columns for column in ACCEL_COLUMNS):
        groups.append("accelerometer")
    if any(column in columns for column in GYRO_COLUMNS):
        groups.append("gyroscope")
    if any(column in columns for column in PPG_COLUMNS):
        groups.append("ppg")
    if "Heart_Rate" in columns:
        groups.append("heart_rate")
    if "HeadBandOn" in columns or any(column in columns for column in HSI_COLUMNS):
        groups.append("hsi/contact")
    if "Battery" in columns:
        groups.append("battery")
    return groups


def _read_mindmonitor_table(path: Path, source_label: str) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path), source_label
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and not Path(name).name.startswith("._")
                and Path(name).suffix.lower() == ".csv"
            ]
            if len(csv_names) != 1:
                raise ValueError(f"Expected exactly one CSV in ZIP archive, found {len(csv_names)}.")
            csv_name = csv_names[0]
            with archive.open(csv_name) as csv_file:
                payload = csv_file.read()
        return _read_csv(BytesIO(payload)), f"{source_label}::{csv_name}"
    raise ValueError(f"Unsupported MindMonitor file type: {suffix}")


def _read_csv(path_or_buffer: str | Path | BinaryIO | BytesIO) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer, skipinitialspace=True, low_memory=False)


def _clean_column_name(column: object) -> str:
    return str(column).strip().lstrip("\ufeff")


def _parse_timestamps(df: pd.DataFrame) -> pd.Series:
    if "TimeStamp" not in df:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df["TimeStamp"], errors="coerce")


def _session_minutes(start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    if pd.isna(start) or pd.isna(end):
        return None
    minutes = (end - start).total_seconds() / 60.0
    if minutes < 0:
        return None
    return round(minutes, 3)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in df]
    if not present:
        return pd.DataFrame(index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce")


def _valid_any_numeric_rows(df: pd.DataFrame, columns: list[str]) -> int:
    numeric = _numeric_frame(df, columns)
    if numeric.empty:
        return 0
    return int(numeric.notna().any(axis=1).sum())


def _complete_numeric_rows(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = _numeric_frame(df, columns)
    if len(numeric.columns) != len(columns):
        return pd.Series(False, index=df.index)
    return numeric.notna().all(axis=1)


def _magnitude(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = _numeric_frame(df, columns)
    if len(numeric.columns) != len(columns):
        return pd.Series(dtype="float64")
    complete = numeric.dropna()
    if complete.empty:
        return pd.Series(dtype="float64")
    squared = complete.pow(2).sum(axis=1)
    return squared.map(math.sqrt)


def _band_mean(df: pd.DataFrame, band: str) -> float | None:
    columns = [f"{band}_{electrode}" for electrode in ELECTRODES if f"{band}_{electrode}" in df]
    if not columns:
        return None
    numeric = _numeric_frame(df, columns)
    values = pd.Series(numeric.to_numpy().ravel()).dropna()
    values = values[values != 0]
    return _mean(values)


def _headband_on_fraction(df: pd.DataFrame) -> float | None:
    if "HeadBandOn" not in df:
        return None
    values = df["HeadBandOn"].map(_headband_value)
    values = values.dropna()
    if values.empty:
        return None
    return float(values.mean())


def _headband_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "on"}:
        return 1.0
    if text in {"false", "no", "off"}:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        return None
    return 1.0 if numeric > 0 else 0.0


def _mean(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else float(values.mean())


def _median(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else float(values.median())


def _quantile(values: pd.Series, quantile: float) -> float | None:
    values = values.dropna()
    return None if values.empty else float(values.quantile(quantile))


def _min(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else float(values.min())


def _max(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else float(values.max())


def _store_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
        return round(value, 3)
    return value


def _build_notes(
    *,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
    columns_present: list[str],
    valid_eeg_rows: int,
    valid_motion_rows: int,
    valid_ppg_rows: int,
) -> str:
    start_text = _format_timestamp(session_start) or "unknown"
    end_text = _format_timestamp(session_end) or "unknown"
    return (
        f"MindMonitor CSV sensor log; session start={start_text}; session end={end_text}; "
        f"columns present={', '.join(columns_present) if columns_present else '(none)'}; "
        f"valid EEG rows={valid_eeg_rows}; valid motion rows={valid_motion_rows}; "
        f"valid PPG rows={valid_ppg_rows}; no sleep staging performed."
    )


def _format_timestamp(value: pd.Timestamp) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat(sep=" ")
