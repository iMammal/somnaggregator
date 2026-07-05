"""Muse sleep EEG screenshot/PDF summary extraction."""

from __future__ import annotations

from pathlib import Path

from .common import parse_wellness_text, read_source_text, source_file_label


def parse_muse_text(
    text: str,
    *,
    source_file: str,
    device: str = "Muse",
    extraction_method: str = "parsed_text",
    confidence: str = "high",
    notes: str = "",
    page: int | None = None,
) -> list[dict[str, object]]:
    """Parse Muse sleep summary text."""

    return parse_wellness_text(
        text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        page=page,
    )


def extract_file(path: str | Path, device: str | None = None) -> list[dict[str, object]]:
    """Extract Muse observations from one screenshot or PDF."""

    source_path = Path(path)
    source = read_source_text(source_path)
    return parse_muse_text(
        source.text,
        source_file=source_file_label(source_path),
        device=device or "Muse",
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )
