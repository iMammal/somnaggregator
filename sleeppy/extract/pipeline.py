"""End-to-end extraction pipeline for sample sleep files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sleeppy.quality import write_extraction_outputs
from sleeppy.schema import OBSERVATION_COLUMNS, ensure_observations_frame

from . import mind_monitor, muse, oscar, oura, samsung
from .common import (
    SUPPORTED_EXTENSIONS,
    check_ocr_environment,
    infer_device_from_path,
    infer_date,
    read_source_text,
    source_file_label,
    extract_pdf_pages_text,
    ocr_pdf_pages_text,
    load_date_mapping,
    observation,
)


DEVICE_FOLDERS = {
    "oura4": ("Oura Ring 4 finger", oura.parse_oura_text),
    "oura3": ("Oura Ring 3 toe", oura.parse_oura_text),
    "samsung_watch": ("Samsung Watch / SleepWatch", samsung.parse_samsung_text),
    "muse": ("Muse", muse.parse_muse_text),
    "oscar": ("ResMed AirSense 11", oscar.parse_oscar_text),
}
MINDMONITOR_REPORT_PREFIX = "MINDMONITOR_REPORT:"


def run_sample_extraction(
    raw_samples_dir: str | Path = "data/raw/samples",
    processed_dir: str | Path = "data/processed",
    outputs_dir: str | Path = "outputs",
    include_legacy_raw: bool = True,
    only_folders: list[str] | None = None,
    only_files: list[str] | None = None,
    max_files: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Extract observations from sample folders and write normalized outputs."""
    from .common import set_verbose
    set_verbose(verbose)

    raw_samples_path = Path(raw_samples_dir)
    processed_path = Path(processed_dir)
    outputs_path = Path(outputs_dir)
    project_root = Path.cwd()
    
    import time
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    last_time = time.time()
    
    def log_time(phase):
        nonlocal last_time
        now = time.time()
        print(f"DEBUG: {phase} took {now - last_time:.2f}s")
        last_time = now
        return now

    def get_path_string(path: Path) -> str:
        if verbose:
            return str(path.absolute())
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)
    observations: list[dict[str, object]] = []
    env = check_ocr_environment()
    report_lines: list[str] = [
        (
            "OCR environment: "
            f"python={env['python_executable']}; "
            f"pillow={env['pillow_installed']}; "
            f"pytesseract={env['pytesseract_installed']}; "
            f"tesseract_cmd={env['tesseract_cmd']}; "
            f"image_ocr_ready={env['image_ocr_ready']}; "
            f"pymupdf={env['pymupdf_installed']}; "
            f"notes={env['notes']}"
        )
    ]
    
    last_time = log_time("Initialization")

    processed_files_count = 0

    for folder_name, (device, parser) in DEVICE_FOLDERS.items():
        if only_folders and folder_name not in only_folders:
            continue
        
        folder = raw_samples_path / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        files = _supported_files(folder)
        if only_files:
            files = [f for f in files if f.name in only_files or str(f) in only_files]
        
        if not files:
            report_lines.append(f"{get_path_string(folder)}: no sample files found.")
            continue
        
        last_time = log_time(f"Scanning files in {folder_name}")
        
        extracted_count = 0
        total_files = len(files)
        
        for path in files:
            if max_files and processed_files_count >= max_files:
                break
            
            rows, source_note, diagnostics = _extract_with_details(path, device, parser)
            observations.extend(rows)
            if len(rows) > 0:
                extracted_count += 1
            processed_files_count += 1
            
            if verbose:
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for {device}; {source_note}")
                report_lines.append(_format_file_diagnostic_line(path, device, diagnostics, get_path_string))
            elif diagnostics["inferred_date"] == "2026-07-07" or len(rows) == 0:
                report_lines.append(_format_file_diagnostic_line(path, device, diagnostics, get_path_string))
        
        last_time = log_time(f"Parsing files in {folder_name}")
        
        if not verbose:
            if extracted_count == 0:
                report_lines.append(f"{get_path_string(folder)}: {device} files detected, but no supported metrics were extracted.")
            else:
                report_lines.append(f"{get_path_string(folder)}: {device} files detected; {extracted_count} of {total_files} produced values.")
        
        if max_files and processed_files_count >= max_files:
            break

    if not only_folders or "mind_monitor" in only_folders:
        folder = raw_samples_path / "mind_monitor"
        folder.mkdir(parents=True, exist_ok=True)
        files = mind_monitor.find_files(folder)
        if only_files:
            files = [f for f in files if f.name in only_files or str(f) in only_files]

        mindmonitor_report: dict[str, object] = {
            "files_detected": len(files),
            "rows_parsed": 0,
            "observations_extracted": 0,
            "channel_groups": [],
            "sessions": [],
        }

        if not files:
            report_lines.append(f"{get_path_string(folder)}: no MindMonitor CSV files found.")
        else:
            last_time = log_time("Scanning files in mind_monitor")
            extracted_count = 0
            channel_groups: set[str] = set()
            sessions: list[dict[str, object]] = []

            for path in files:
                if max_files and processed_files_count >= max_files:
                    break

                rows, diagnostics = mind_monitor.extract_file_with_details(path)
                observations.extend(rows)
                processed_files_count += 1
                if rows:
                    extracted_count += 1

                diagnostic_report = diagnostics.to_report_dict()
                mindmonitor_report["rows_parsed"] = int(mindmonitor_report["rows_parsed"]) + int(diagnostic_report["rows_parsed"])
                mindmonitor_report["observations_extracted"] = int(mindmonitor_report["observations_extracted"]) + int(
                    diagnostic_report["observations_extracted"]
                )
                channel_groups.update(str(group) for group in diagnostic_report["channel_groups"])
                sessions.append(
                    {
                        "source_file": get_path_string(path),
                        "rows_parsed": diagnostic_report["rows_parsed"],
                        "observations_extracted": diagnostic_report["observations_extracted"],
                        "session_start": diagnostic_report["session_start"],
                        "session_end": diagnostic_report["session_end"],
                        "session_minutes": diagnostic_report["session_minutes"],
                        "crossed_midnight": diagnostic_report["crossed_midnight"],
                        "stopped_before_morning": diagnostic_report["stopped_before_morning"],
                        "gap_count_gt_5s": diagnostic_report["gap_count_gt_5s"],
                        "max_gap_seconds": diagnostic_report["max_gap_seconds"],
                        "battery_min": diagnostic_report["battery_min"],
                        "battery_max": diagnostic_report["battery_max"],
                        "valid_eeg_rows": diagnostic_report["valid_eeg_rows"],
                        "valid_motion_rows": diagnostic_report["valid_motion_rows"],
                        "valid_ppg_rows": diagnostic_report["valid_ppg_rows"],
                        "error": diagnostic_report["error"],
                    }
                )

                if verbose:
                    report_lines.append(
                        f"{get_path_string(path)}: extracted {len(rows)} MindMonitor values; "
                        f"rows={diagnostic_report['rows_parsed']}; "
                        f"session_minutes={diagnostic_report['session_minutes']}; "
                        f"crossed_midnight={diagnostic_report['crossed_midnight']}; "
                        f"stopped_before_morning={diagnostic_report['stopped_before_morning']}; "
                        f"channel_groups={', '.join(diagnostic_report['channel_groups']) or '(none)'}"
                    )

            mindmonitor_report["channel_groups"] = sorted(channel_groups)
            mindmonitor_report["sessions"] = sessions
            if not verbose:
                report_lines.append(
                    f"{get_path_string(folder)}: MindMonitor files detected; "
                    f"{extracted_count} of {len(files)} produced values."
                )
            last_time = log_time("Parsing files in mind_monitor")

        report_lines.append(f"{MINDMONITOR_REPORT_PREFIX}{json.dumps(mindmonitor_report, sort_keys=True)}")

    if include_legacy_raw and not only_folders:
        legacy_files = _legacy_raw_files(raw_samples_path.parent)
        if only_files:
            legacy_files = [f for f in legacy_files if f.name in only_files or str(f) in only_files]
            
        for path in legacy_files:
            if max_files and processed_files_count >= max_files:
                break
            device = infer_device_from_path(path)
            parser = _parser_for_device(device)
            rows, source_note, diagnostics = _extract_with_details(path, device, parser)
            observations.extend(rows)
            processed_files_count += 1
            if verbose:
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for inferred device {device}; {source_note}")
                report_lines.append(_format_file_diagnostic_line(path, device, diagnostics, get_path_string))
            else:
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for inferred device {device}.")
                if diagnostics["inferred_date"] == "2026-07-07" or len(rows) == 0:
                    report_lines.append(_format_file_diagnostic_line(path, device, diagnostics, get_path_string))

    mixed_folder = raw_samples_path / "mixed"
    if mixed_folder.exists():
        if not only_folders or "mixed" in only_folders:
            mixed_files = _supported_files(mixed_folder)
            if only_files:
                mixed_files = [f for f in mixed_files if f.name in only_files or str(f) in only_files]

            for path in mixed_files:
                if max_files and processed_files_count >= max_files:
                    break
                print(f"Processing mixed file: {path.name}")
                rows = _extract_mixed_files(path, report_lines)
                observations.extend(rows)
                processed_files_count += 1
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for mixed device(s).")

    _add_mindmonitor_expected_sleep_window_coverage(observations)
    long_df = ensure_observations_frame(pd.DataFrame(observations, columns=OBSERVATION_COLUMNS))
    
    from .common import CACHE_STATS
    CACHE_STATS.report()

    return write_extraction_outputs(long_df, processed_path, outputs_path, report_lines)


def _add_mindmonitor_expected_sleep_window_coverage(observations: list[dict[str, object]]) -> None:
    """Add optional MindMonitor coverage pct when same-night bed-window metrics exist."""

    if not observations:
        return
    by_night: dict[str, list[dict[str, object]]] = {}
    for row in observations:
        night_date = row.get("night_date")
        if night_date is None or pd.isna(night_date):
            continue
        by_night.setdefault(str(night_date), []).append(row)

    for night_date, rows in by_night.items():
        expected_minutes = [
            _numeric_observation_value(row)
            for row in rows
            if row.get("metric") == "time_in_bed_minutes"
            and str(row.get("device", "")).startswith(("Oura", "Samsung"))
        ]
        expected_minutes = [value for value in expected_minutes if value is not None and value > 0]
        if not expected_minutes:
            continue
        expected = max(expected_minutes)
        mindmonitor_session_rows = [
            row
            for row in rows
            if row.get("device") == mind_monitor.DEVICE_NAME
            and row.get("metric") == "mindmonitor_session_minutes"
        ]
        for session_row in mindmonitor_session_rows:
            session_minutes = _numeric_observation_value(session_row)
            if session_minutes is None:
                continue
            coverage = round((session_minutes / expected) * 100.0, 3)
            observations.append(
                observation(
                    date=night_date,
                    device=mind_monitor.DEVICE_NAME,
                    metric="mindmonitor_expected_sleep_window_coverage_pct",
                    value=coverage,
                    unit="pct",
                    source_file=str(session_row.get("source_file", "")),
                    extraction_method="derived",
                    confidence="medium",
                    notes=(
                        "Derived from MindMonitor session_minutes divided by same-night "
                        f"Oura/Samsung time_in_bed_minutes ({expected:g}); no sleep staging performed."
                    ),
                )
            )


def _numeric_observation_value(row: dict[str, object]) -> float | None:
    value = row.get("value")
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_mixed_files(path: Path, report_lines: list[str]) -> list[dict[str, object]]:
    observations = []
    print(f"DEBUG: Extracting mixed file: {path.name}")
    
    # Simple profiling
    import time
    start_page_time = time.time()
    
    text_pages = extract_pdf_pages_text(path)
    if not any(text.strip() for text in text_pages):
        text_pages = ocr_pdf_pages_text(path)
    
    print(f"DEBUG: PDF page extraction for {path.name} took {time.time() - start_page_time:.2f}s")
    
    mapping = load_date_mapping()
    source_label = source_file_label(path)
    
    # Try mapping by relative path as in common.py
    try:
        rel_path = path.resolve().relative_to(Path.cwd().resolve())
        source_key = rel_path.as_posix()
    except ValueError:
        source_key = source_label

    for i, text in enumerate(text_pages):
        page_num = i + 1
        device = None
        
        # Check mapping for this page
        if source_key in mapping and page_num in mapping[source_key]:
            device = mapping[source_key][page_num].get("device")
        
        # If device not in mapping, heuristics
        if not device:
            text_lower = text.lower()
            if "oura" in text_lower:
                device = "Oura Ring"
            elif "muse" in text_lower:
                device = "Muse"
            elif "samsung" in text_lower or "health" in text_lower:
                device = "Samsung Watch"
        
        if device:
            parser = _parser_for_device(device)
            # Use original parser call to get observations, 
            # need to check its signature. It takes (text, source_file=..., device=..., ...)
            rows = parser(
                text,
                source_file=f"{source_label} (page {page_num})",
                device=device,
                extraction_method="parsed_text" if text else "ocr",
                confidence="medium",
                notes=f"Extracted from page {page_num}",
                page=page_num,
            )
            observations.extend(rows)
            report_lines.append(f"  Page {page_num}: detected {device}, extracted {len(rows)} values (text length {len(text)})")
        else:
            report_lines.append(f"  Page {page_num}: no device detected (text length {len(text)})")
            
    return observations


def _supported_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _legacy_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    sample_root = (raw_dir / "samples").resolve()
    files = []
    for path in raw_dir.iterdir():
        if not path.is_file() or path.name.startswith("._") or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(sample_root)
        except ValueError:
            files.append(path)
    return sorted(files)


def _extract_with_details(path: Path, device: str, parser) -> tuple[list[dict[str, object]], str, dict[str, object]]:
    source = read_source_text(path)
    rows = parser(
        source.text,
        source_file=source_file_label(path),
        device=device,
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )
    diagnostics = {
        "source_text": source.text,
        "source_text_length": len(source.text),
        "page_count": 0,
        "page_text_lengths": [],
        "inferred_date": infer_date(source.text, path) or "undated",
        "metric_names": sorted({str(row.get("metric")) for row in rows if row.get("metric")}),
        "row_count": len(rows),
        "reason": _zero_row_reason(device, source.text, path, len(rows)),
    }
    if path.suffix.lower() == ".pdf":
        page_texts = extract_pdf_pages_text(path)
        diagnostics["page_count"] = len(page_texts)
        diagnostics["page_text_lengths"] = [len(text) for text in page_texts]
    return rows, source.notes, diagnostics


def _format_file_diagnostic_line(
    path: Path,
    device: str,
    diagnostics: dict[str, object],
    get_path_string,
) -> str:
    metric_names = diagnostics.get("metric_names") or []
    metric_text = ", ".join(metric_names) if metric_names else "(none)"
    page_lengths = diagnostics.get("page_text_lengths") or []
    detail_parts = [
        f"{get_path_string(path)}:",
        f"device={device}",
        f"date={diagnostics.get('inferred_date')}",
        f"pages={diagnostics.get('page_count')}",
        f"page_text_lengths={page_lengths}",
        f"cache_text_length={diagnostics.get('source_text_length')}",
        f"rows={diagnostics.get('row_count')}",
        f"metrics={metric_text}",
    ]
    reason = diagnostics.get("reason")
    if reason:
        detail_parts.append(f"reason={reason}")
    if device.startswith("Samsung"):
        from . import samsung as samsung_module

        snippets = samsung_module.diagnostic_summary(str(diagnostics.get("source_text", "")))
        detail_parts.extend(
            [
                f"sleep_duration={snippets.get('sleep_duration')}",
                f"stages={snippets.get('stages')}",
                f"blood_oxygen={snippets.get('blood_oxygen')}",
                f"heart_rate={snippets.get('heart_rate')}",
                f"respiratory_rate={snippets.get('respiratory_rate')}",
            ]
        )
    elif device == "Muse":
        from . import muse as muse_module

        snippets = muse_module.diagnostic_summary(str(diagnostics.get("source_text", "")))
        detail_parts.extend(
            [
                f"session={snippets.get('session')}",
                f"stages={snippets.get('stages')}",
                f"heart_rate={snippets.get('heart_rate')}",
                f"respiratory_rate={snippets.get('respiratory_rate')}",
            ]
        )
    elif device.startswith("Oura"):
        import importlib
        from . import oura as oura_module
        importlib.reload(oura_module)

        snippets = oura_module.diagnostic_summary(str(diagnostics.get("source_text", "")))
        detail_parts.extend(
            [
                f"score_card={snippets.get('score_card')}",
                f"details={snippets.get('details')}",
                f"hrv={snippets.get('hrv')}",
            ]
        )
    return "; ".join(detail_parts)


def _zero_row_reason(device: str, text: str, path: Path, row_count: int) -> str:
    if row_count > 0:
        return ""
    if device == "Muse":
        from . import muse as muse_module

        reason = muse_module.diagnostic_summary(text).get("reason")
        if reason:
            return str(reason)
        return "Muse OCR text is too sparse for conservative extraction."
    if device.startswith("Samsung"):
        return "Samsung OCR text did not expose a conservative sleep block."
    if device.startswith("Oura"):
        return "Oura OCR text did not expose a recoverable sleep card."
    if path.suffix.lower() == ".pdf":
        return "PDF text extraction returned no usable text."
    return "No usable OCR text was available."


def _parser_for_device(device: str):
    if device.startswith("Oura"):
        return oura.parse_oura_text
    if device.startswith("Samsung"):
        return samsung.parse_samsung_text
    if device == "Muse":
        return muse.parse_muse_text
    if device.startswith("ResMed"):
        return oscar.parse_oscar_text
    return samsung.parse_samsung_text
