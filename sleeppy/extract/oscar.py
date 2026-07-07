"""OSCAR/SleepScope CPAP PDF or screenshot summary extraction."""

from __future__ import annotations

from pathlib import Path

from .common import parse_cpap_text, read_source_text, source_file_label


def parse_oscar_text(
    text: str,
    *,
    source_file: str,
    device: str = "ResMed AirSense 11",
    extraction_method: str = "parsed_text",
    confidence: str = "high",
    notes: str = "",
    page: int | None = None,
    **kwargs,
) -> list[dict[str, object]]:
    """Parse OSCAR/SleepScope CPAP summary text."""

    return parse_cpap_text(
        text,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        page=page,
    )


def extract_file(path: str | Path, device: str | None = None) -> list[dict[str, object]]:
    """Extract CPAP observations from one screenshot or PDF."""

    source_path = Path(path)
    source = read_source_text(source_path)
    return parse_oscar_text(
        source.text,
        source_file=source_file_label(source_path),
        device=device or "ResMed AirSense 11",
        extraction_method=source.extraction_method,
        confidence=source.confidence,
        notes=source.notes,
    )
