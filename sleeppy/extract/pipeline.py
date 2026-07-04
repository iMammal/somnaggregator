"""End-to-end extraction pipeline for sample sleep files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sleeppy.quality import write_extraction_outputs
from sleeppy.schema import OBSERVATION_COLUMNS, ensure_observations_frame

from . import muse, oscar, oura, samsung
from .common import SUPPORTED_EXTENSIONS, check_ocr_environment, infer_device_from_path, read_source_text, source_file_label


DEVICE_FOLDERS = {
    "oura4": ("Oura Ring 4 finger", oura.parse_oura_text),
    "oura3": ("Oura Ring 3 toe", oura.parse_oura_text),
    "samsung_watch": ("Samsung Watch / SleepWatch", samsung.parse_samsung_text),
    "muse": ("Muse", muse.parse_muse_text),
    "oscar": ("ResMed AirSense 11", oscar.parse_oscar_text),
}


def run_sample_extraction(
    raw_samples_dir: str | Path = "data/raw/samples",
    processed_dir: str | Path = "data/processed",
    outputs_dir: str | Path = "outputs",
    include_legacy_raw: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Extract observations from sample folders and write normalized outputs."""

    raw_samples_path = Path(raw_samples_dir)
    processed_path = Path(processed_dir)
    outputs_path = Path(outputs_dir)
    project_root = Path.cwd()

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

    for folder_name, (device, parser) in DEVICE_FOLDERS.items():
        folder = raw_samples_path / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        files = _supported_files(folder)
        if not files:
            report_lines.append(f"{get_path_string(folder)}: no sample files found.")
            continue
        extracted_count = 0
        total_files = len(files)
        
        for path in files:
            rows, source_note = _extract_with_details(path, device, parser)
            observations.extend(rows)
            if len(rows) > 0:
                extracted_count += 1
            
            if verbose:
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for {device}; {source_note}")
        
        if not verbose:
            if extracted_count == 0:
                report_lines.append(f"{get_path_string(folder)}: {device} files detected, but no supported metrics were extracted.")
            else:
                report_lines.append(f"{get_path_string(folder)}: {device} files detected; {extracted_count} of {total_files} produced values.")

    if include_legacy_raw:
        legacy_files = _legacy_raw_files(raw_samples_path.parent)
        for path in legacy_files:
            device = infer_device_from_path(path)
            parser = _parser_for_device(device)
            rows, source_note = _extract_with_details(path, device, parser)
            observations.extend(rows)
            if verbose:
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for inferred device {device}; {source_note}")
            else:
                report_lines.append(f"{get_path_string(path)}: extracted {len(rows)} values for inferred device {device}.")

    long_df = ensure_observations_frame(pd.DataFrame(observations, columns=OBSERVATION_COLUMNS))
    return write_extraction_outputs(long_df, processed_path, outputs_path, report_lines)


def _supported_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _legacy_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    sample_root = (raw_dir / "samples").resolve()
    files = []
    for path in raw_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(sample_root)
        except ValueError:
            files.append(path)
    return sorted(files)


def _extract_with_details(path: Path, device: str, parser) -> tuple[list[dict[str, object]], str]:
    source = read_source_text(path)
    rows = parser(
        source.text,
        source_file=source_file_label(path),
        device=device,
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )
    return rows, source.notes


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
