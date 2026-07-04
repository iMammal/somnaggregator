"""Oura screenshot/PDF summary extraction."""

from __future__ import annotations

from pathlib import Path

from .common import infer_device_from_path, parse_wellness_text, read_source_text, source_file_label


def parse_oura_text(
    text: str,
    *,
    source_file: str,
    device: str = "Oura Ring",
    extraction_method: str = "parsed_text",
    confidence: str = "high",
    notes: str = "",
) -> list[dict[str, object]]:
    """Parse Oura sleep summary text into long-form observations."""

    return parse_wellness_text(
        text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
    )


def extract_file(path: str | Path, device: str | None = None) -> list[dict[str, object]]:
    """Extract Oura observations from one screenshot or PDF."""

    source_path = Path(path)
    source = read_source_text(source_path)
    inferred_device = device or infer_device_from_path(source_path)
    if inferred_device == "Unknown device":
        inferred_device = "Oura Ring"
    return parse_oura_text(
        source.text,
        source_file=source_file_label(source_path),
        device=inferred_device,
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )
