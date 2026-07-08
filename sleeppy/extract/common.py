"""Shared text extraction and regex parsing helpers."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Callable


def _get_file_hash(path: Path) -> str:
    """Return a SHA-256 hash of the file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_cache_dir() -> Path:
    cache_dir = Path("data/interim/ocr_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VERBOSE = False

def set_verbose(v: bool):
    global VERBOSE
    VERBOSE = v

class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.start_time = datetime.now()

    def hit(self):
        self.hits += 1

    def miss(self):
        self.misses += 1

    def report(self):
        elapsed = datetime.now() - self.start_time
        print(f"Extraction summary: {self.hits} cache hits, {self.misses} cache misses.")
        print(f"Elapsed time: {elapsed}")

CACHE_STATS = CacheStats()
TESSERACT_CANDIDATES = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]
METRIC_BOUNDS = {
    "sleep_score": (0, 100),
    "total_sleep_minutes": (15, 900),
    "time_in_bed_minutes": (15, 1200),
    "sleep_efficiency_pct": (50, 100),
    "awake_minutes": (0, 480),
    "rem_minutes": (0, 480),
    "light_minutes": (0, 720),
    "deep_minutes": (0, 360),
    "min_hr_bpm": (25, 120),
    "avg_hr_bpm": (25, 140),
    "avg_hrv_ms": (5, 250),
    "max_hrv_ms": (5, 400),
    "hrv_balance_score": (0, 100),
    "avg_spo2_pct": (70, 100),
    "min_spo2_pct": (50, 100),
    "respiratory_rate_bpm": (5, 40),
    "temperature_deviation_c": (-5, 5),
    "readiness_score": (0, 100),
    "cpap_ahi": (0, 120),
    "cpap_cai": (0, 120),
    "cpap_oai": (0, 120),
    "cpap_pressure": (0, 30),
    "cpap_leak_rate": (0, 200),
    "cpap_usage_hours": (0, 24),
}


@dataclass
class SourceText:
    """Text extracted from one source file plus provenance."""

    text: str
    extraction_method: str
    confidence: str
    notes: str


def source_file_label(path: Path) -> str:
    """Return a stable source-file label relative to the project when possible."""

    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def read_source_text(path: Path) -> SourceText:
    """Extract text from a PDF or image, preferring parsed PDF text before OCR."""
    
    file_hash = _get_file_hash(path)
    
    suffix = path.suffix.lower()
    
    # Try cache first (simplified cache: assume single-page or cache key includes page if needed, 
    # but here we cache whole file text if possible)
    cache_file = _get_cache_dir() / f"{file_hash}.txt"
    if cache_file.exists():
        CACHE_STATS.hit()
        if VERBOSE:
            print(f"DEBUG: Cache hit for {path}")
        return SourceText(text=cache_file.read_text(encoding="utf-8"), extraction_method="cache", confidence="high", notes="cached")
    
    CACHE_STATS.miss()
    if VERBOSE:
        print(f"DEBUG: Cache miss for {path}")

    if suffix == ".pdf":
        text, notes = extract_pdf_text(path)
        if text.strip():
            st = SourceText(text=text, extraction_method="parsed_text", confidence="high", notes=notes)
            cache_file.write_text(st.text, encoding="utf-8")
            return st

        ocr_text, ocr_notes = ocr_pdf_pages(path)
        if ocr_text.strip():
            st = SourceText(
                text=ocr_text,
                extraction_method="ocr",
                confidence="medium",
                notes=_join_notes(notes, ocr_notes),
            )
            cache_file.write_text(st.text, encoding="utf-8")
            return st
        return SourceText(text="", extraction_method="manual", confidence="low", notes=_join_notes(notes, ocr_notes))

    if suffix in IMAGE_EXTENSIONS:
        text, notes = ocr_image(path)
        if text.strip():
            st = SourceText(text=text, extraction_method="ocr", confidence="medium", notes=notes)
            cache_file.write_text(st.text, encoding="utf-8")
            return st
        return SourceText(text="", extraction_method="manual", confidence="low", notes=notes)

    return SourceText(text="", extraction_method="manual", confidence="low", notes=f"Unsupported file type: {suffix}")


def check_ocr_environment() -> dict[str, object]:
    """Return OCR/PDF dependency status for the active Python environment."""

    tesseract_cmd = find_tesseract_executable()
    status: dict[str, object] = {
        "python_executable": sys.executable,
        "pillow_installed": find_spec("PIL") is not None,
        "pytesseract_installed": find_spec("pytesseract") is not None,
        "pymupdf_installed": find_spec("fitz") is not None,
        "tesseract_cmd": str(tesseract_cmd) if tesseract_cmd else None,
        "tesseract_version": None,
        "image_ocr_ready": False,
        "pdf_text_ready": find_spec("fitz") is not None,
        "notes": "",
    }

    notes: list[str] = []
    if not status["pillow_installed"]:
        notes.append("Pillow is not installed in this Python environment.")
    if not status["pytesseract_installed"]:
        notes.append("pytesseract is not installed in this Python environment.")

    if status["pillow_installed"] and status["pytesseract_installed"]:
        try:
            import pytesseract

            if tesseract_cmd is not None:
                pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
            version = pytesseract.get_tesseract_version()
            status["tesseract_version"] = str(version)
            status["image_ocr_ready"] = True
        except (Exception, SystemExit) as exc:  # pragma: no cover - depends on local executable
            notes.append(f"Tesseract executable is not usable from this environment: {exc}")

    if not status["pymupdf_installed"]:
        notes.append("PyMuPDF is not installed; PDF parsed text extraction is unavailable.")

    status["notes"] = " ".join(notes) if notes else "OCR/PDF dependencies look ready."
    return status


def find_tesseract_executable() -> Path | None:
    """Find Tesseract from PATH, env var, or standard Windows install paths."""

    import os

    env_value = os.environ.get("TESSERACT_CMD")
    if env_value:
        env_path = Path(env_value)
        if env_path.exists():
            return env_path

    path_value = shutil.which("tesseract")
    if path_value:
        return Path(path_value)

    for candidate in TESSERACT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def configure_pytesseract() -> tuple[bool, str]:
    """Configure pytesseract with a discovered executable if possible."""

    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract is not installed."

    tesseract_cmd = find_tesseract_executable()
    if tesseract_cmd is None:
        return False, "Tesseract executable was not found on PATH or in standard Windows install locations."

    pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
    return True, f"Using Tesseract executable at {tesseract_cmd}."


def extract_pdf_text(path: Path) -> tuple[str, str]:
    """Extract selectable text from a PDF with PyMuPDF if installed."""

    try:
        import fitz  # type: ignore
    except ImportError:
        return "", "PyMuPDF is not installed; PDF parsed-text extraction skipped."

    try:
        with fitz.open(path) as document:
            text = "\n".join(page.get_text("text") for page in document)
        return text, "Extracted PDF text with PyMuPDF."
    except Exception as exc:  # pragma: no cover - defensive around external files
        return "", f"PyMuPDF failed: {exc}"


def get_pdf_page_cache_path(file_hash: str, page_num: int) -> Path:
    return _get_cache_dir() / f"{file_hash}_page_{page_num}.txt"

def extract_pdf_pages_text(path: Path) -> list[str]:
    """Extract selectable text from a PDF page-by-page, with caching."""
    file_hash = _get_file_hash(path)
    try:
        import fitz  # type: ignore
        with fitz.open(path) as document:
            text_pages = []
            for i, page in enumerate(document):
                page_num = i + 1
                cache_path = get_pdf_page_cache_path(file_hash, page_num)
                if cache_path.exists():
                    CACHE_STATS.hit()
                    if VERBOSE:
                        print(f"DEBUG: PDF page cache hit: {path.name} page {page_num}")
                    text_pages.append(cache_path.read_text(encoding="utf-8"))
                else:
                    CACHE_STATS.miss()
                    if VERBOSE:
                        print(f"DEBUG: PDF page cache miss: {path.name} page {page_num}")
                    text = page.get_text("text")
                    cache_path.write_text(text, encoding="utf-8")
                    text_pages.append(text)
            return text_pages
    except Exception:
        return []


def ocr_image(path: Path) -> tuple[str, str]:
    """OCR one image with pytesseract when available."""

    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return "", "pytesseract/Pillow is not installed; image OCR skipped."

    ready, note = configure_pytesseract()
    if not ready:
        return "", note

    try:
        return pytesseract.image_to_string(Image.open(path)), f"OCR extracted with pytesseract. {note}"
    except Exception as exc:  # pragma: no cover - depends on local OCR install
        return "", f"pytesseract failed: {exc}"


def ocr_pdf_pages(path: Path, dpi: int = 180) -> tuple[str, str]:
    """Render PDF pages with PyMuPDF and OCR them if all optional dependencies exist."""

    try:
        import fitz  # type: ignore
        from PIL import Image
        import pytesseract
    except ImportError:
        return "", "PyMuPDF, Pillow, or pytesseract is not installed; PDF OCR skipped."

    ready, note = configure_pytesseract()
    if not ready:
        return "", note

    try:
        text_parts = []
        with fitz.open(path) as document:
            for page in document:
                pixmap = page.get_pixmap(dpi=dpi)
                mode = "RGB" if pixmap.alpha == 0 else "RGBA"
                image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
                text_parts.append(pytesseract.image_to_string(image))
        return "\n".join(text_parts), f"Rendered PDF pages with PyMuPDF and OCRed with pytesseract. {note}"
    except Exception as exc:  # pragma: no cover - depends on external files
        return "", f"PDF OCR failed: {exc}"


def ocr_pdf_pages_text(path: Path, dpi: int = 180) -> list[str]:
    """Render PDF pages with PyMuPDF and OCR them if all optional dependencies exist, with caching."""
    file_hash = _get_file_hash(path)
    try:
        import fitz  # type: ignore
        from PIL import Image
        import pytesseract
        
        with fitz.open(path) as document:
            text_pages = []
            for i, page in enumerate(document):
                page_num = i + 1
                # Use a specific cache path for OCR text
                cache_path = get_pdf_page_cache_path(file_hash, page_num)
                # Need a separate cache path for OCR specifically, maybe _ocr?
                # Actually, requirement says: "If cached text exists for a PDF page... do not OCR it"
                # So if `get_pdf_page_cache_path` exists, we can use it.
                # But wait, `ocr_pdf_pages_text` is ONLY called if `extract_pdf_pages_text` (parsed text) failed to find text.
                # So we should be able to reuse the same cache key for "PDF text found for this page".
                # But wait, if I had empty text before, `extract_pdf_pages_text` would have created an empty file in cache.
                # I need to distinguish between empty parsed text and filled OCR text.
                
                # Let's use a different cache key for OCR'ed text: f"{file_hash}_ocr_page_{page_num}.txt"
                ocr_cache_path = _get_cache_dir() / f"{file_hash}_ocr_page_{page_num}.txt"
                
                if ocr_cache_path.exists():
                    text = ocr_cache_path.read_text(encoding="utf-8")
                    if text.strip():
                        CACHE_STATS.hit()
                        if VERBOSE:
                            print(f"PDF page OCR cache hit: {path.name} page {page_num}")
                        text_pages.append(text)
                        continue
                    else:
                        # File exists but is empty, treat as miss and re-OCR
                        if VERBOSE:
                            print(f"PDF page OCR cache empty, treating as miss: {path.name} page {page_num}")
                
                CACHE_STATS.miss()
                if VERBOSE:
                    print(f"PDF page OCR cache miss: {path.name} page {page_num}")
                
                pixmap = page.get_pixmap(dpi=dpi)
                mode = "RGB" if pixmap.alpha == 0 else "RGBA"
                image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
                text = pytesseract.image_to_string(image)
                ocr_cache_path.write_text(text, encoding="utf-8")
                text_pages.append(text)
            return text_pages
    except Exception:
        return []


def parse_duration_to_minutes(text: str) -> int | None:
    """Parse durations such as '6h 52m', '1 hours, 39 minutes, 0 seconds', or '52 min'."""

    value = text.strip().lower()
    value = re.sub(r"\boh", "0h", value)
    value = re.sub(r"\bo(?=\d)", "0", value)
    hms = re.search(
        r"(?:(?P<hours>\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h))?\s*,?\s*"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m))?\s*,?\s*"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s))?",
        value,
    )
    if not hms:
        return None

    hours = float(hms.group("hours") or 0)
    minutes = float(hms.group("minutes") or 0)
    seconds = float(hms.group("seconds") or 0)
    if hours == 0 and minutes == 0 and seconds == 0:
        return None
    return int(round(hours * 60 + minutes + seconds / 60))


def load_date_mapping() -> dict[str, dict[int | None, dict[str, str]]]:
    mapping: dict[str, dict[int | None, dict[str, str]]] = {}
    csv_path = Path("data/manual_date_mapping.csv")
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = Path(row["path"]).as_posix()
                page = int(row.get("page") or 0) if "page" in row else None
                date = row.get("date")
                device = row.get("device")
                
                if path not in mapping:
                    mapping[path] = {}
                mapping[path][page] = {"date": date, "device": device}
                # Support basename mapping too
                basename = Path(row["path"]).name
                if basename not in mapping:
                    mapping[basename] = {}
                mapping[basename][page] = {"date": date, "device": device}
    return mapping


def load_metric_overrides() -> dict[str, dict[int | None, dict[str, str]]]:
    overrides: dict[str, dict[int | None, dict[str, str]]] = {}
    csv_path = Path("data/manual_metric_overrides.csv")
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = Path(row["path"]).as_posix()
                page = int(row.get("page") or 0) if "page" in row else None
                metric = row["metric"]
                value = row["value"]
                
                if path not in overrides:
                    overrides[path] = {}
                if page not in overrides[path]:
                    overrides[path][page] = {}
                overrides[path][page][metric] = value
                
                # Support basename mapping too
                basename = Path(row["path"]).name
                if basename not in overrides:
                    overrides[basename] = {}
                if page not in overrides[basename]:
                    overrides[basename][page] = {}
                overrides[basename][page][metric] = value
    return overrides


def infer_date(text: str, source_file: str | Path, page: int | None = None) -> str | None:
    """Infer a YYYY-MM-DD date from OCR text or source filename."""

    # Check filename in manual mapping first
    path = Path(source_file)
    mapping = load_date_mapping()

    # Try match by exact relative path (normalized to POSIX)
    try:
        rel_path = path.resolve().relative_to(Path.cwd().resolve())
        rel_path_posix = rel_path.as_posix()
        if rel_path_posix in mapping:
            if page in mapping[rel_path_posix]:
                return mapping[rel_path_posix][page]["date"]
            if 0 in mapping[rel_path_posix]:
                return mapping[rel_path_posix][0]["date"]
    except ValueError:
        pass

    # Fallback to name
    name = path.name
    if name in mapping:
        if page in mapping[name]:
            return mapping[name][page]["date"]
        if 0 in mapping[name]:
            return mapping[name][0]["date"]

    patterns = [
        r"\b(?P<year>20\d{2})[-_/\.](?P<month>\d{1,2})[-_/\.](?P<day>\d{1,2})\b",
        r"\b(?P<month>\d{1,2})[-_/\.](?P<day>\d{1,2})[-_/\.](?P<year>20\d{2})\b",
        r"\b(?P<month_name>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<day>\d{1,2}),?\s+(?P<year>20\d{2})\b",
        r"\b(?P<month_name>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<day>\d{1,2})", # Fallback for just Month Day in header
    ]
    # For Oura "May 28, 2026", "May 28" (assuming current year) or "Today"
    # Actually, we need year for YYYY-MM-DD. If year is missing, maybe assume 2026 for now as it's the date in description? Or use current year?
    # Description says: Extract dates from visible app headers when possible, especially Oura “May 28, 2026” / “Today” views.
    # If "Today" is visible, how to know the date? Maybe it's not possible to infer it unless the file has a timestamp.
    # The requirement is just: Extract dates ... especially ... and Samsung filenames.
    
    # For Oura, "May 28, 2026" should be covered by pattern #3 above.
    
    haystacks = [text, str(source_file)]
    if "OSCAR" in str(source_file):
        haystacks = [str(source_file), text]

    for haystack in haystacks:
        for pattern in patterns:
            match = re.search(pattern, haystack, flags=re.IGNORECASE)
            if match:
                inferred = _date_from_match(match)
                if inferred is not None:
                    return inferred

        compact = re.search(r"(?<!\d)(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})(?!\d)", haystack)
        if compact:
            inferred = _date_from_match(compact)
            if inferred is not None:
                return inferred

        # OSCAR-style 2-digit year (MM-DD-YY)
        mm_dd_yy = re.search(r"\b(?P<month>\d{1,2})[-_/\.](?P<day>\d{1,2})[-_/\.](?P<year>\d{2})\b", haystack)
        if mm_dd_yy:
            inferred = _date_from_match(mm_dd_yy)
            if inferred is not None:
                return inferred

    return None


def observation(
    *,
    date: str | None,
    device: str,
    metric: str,
    value: object,
    unit: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
) -> dict[str, object]:
    """Create one normalized observation row."""

    return {
        "night_date": date,
        "device": device,
        "metric": metric,
        "value": value,
        "unit": unit,
        "source_file": source_file,
        "extraction_method": extraction_method,
        "confidence": confidence,
        "notes": notes,
    }


def add_duration_observation(
    rows: list[dict[str, object]],
    text: str,
    *,
    metric: str,
    label_patterns: list[str],
    date: str | None,
    device: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
    unit: str = "minutes",
    value_transform: Callable[[float], float] | None = None,
) -> None:
    """Append a duration observation if one of the label patterns matches."""

    duration_number = r"(?:[oO]?\d+(?:\.\d+)?)"
    for label_pattern in label_patterns:
        match = re.search(
            rf"{label_pattern}\s*[:\-]?\s*(?P<duration>{duration_number}\s*(?:hours?|hrs?|h)(?:\s*,?\s*{duration_number}\s*(?:minutes?|mins?|m))?(?:\s*,?\s*{duration_number}\s*(?:seconds?|secs?|s))?|{duration_number}\s*(?:minutes?|mins?|m))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            minutes = parse_duration_to_minutes(match.group("duration"))
            value = None if minutes is None else float(minutes)
            if value is not None and value_transform is not None:
                value = value_transform(value)
            if value is not None and _within_metric_bounds(metric, float(value)):
                if float(value).is_integer():
                    value = int(value)
                rows.append(
                    observation(
                        date=date,
                        device=device,
                        metric=metric,
                        value=value,
                        unit=unit,
                        source_file=source_file,
                        extraction_method=extraction_method,
                        confidence=confidence,
                        notes=notes,
                    )
                )
                return


def add_number_observation(
    rows: list[dict[str, object]],
    text: str,
    *,
    metric: str,
    unit: str,
    label_patterns: list[str],
    date: str | None,
    device: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
    page: int | None = None,
    value_transform: Callable[[float], float] | None = None,
    flags: int = re.IGNORECASE,
    value_pattern: str = r"[+-]?\d+(?:\.\d+)?",
) -> None:
    """Append a numeric observation if one of the label patterns matches."""

    # Check for manual overrides
    overrides = load_metric_overrides()
    path = Path(source_file)
    path_posix = path.as_posix()
    basename = path.name
    
    lookup_page = page or 0
    
    # Check overrides
    for key in [path_posix, basename]:
        if key in overrides and lookup_page in overrides[key] and metric in overrides[key][lookup_page]:
            value = float(overrides[key][lookup_page][metric])
            if value_transform is not None:
                value = value_transform(value)
            
            if value.is_integer():
                value = int(value)
            rows.append(
                observation(
                    date=date,
                    device=device,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source_file=source_file,
                    extraction_method=f"{extraction_method}-override",
                    confidence="manual",
                    notes=f"{notes} (manual override)",
                )
            )
            return

    # Debug
    # print(f"DEBUG: overrides={overrides}")
    # print(f"DEBUG: key={path_posix} or {basename}, page={page}, metric={metric}")

    for label_pattern in label_patterns:
        match = re.search(
            rf"{label_pattern}\s*[:\-]?\s*(?P<value>{value_pattern})",
            text,
            flags=flags,
        )
        if match:
            value = float(match.group("value"))
            if value_transform is not None:
                value = value_transform(value)
            if not _within_metric_bounds(metric, value):
                if VERBOSE and metric == "avg_spo2_pct":
                    print(f"DEBUG: Found {metric} match '{match.group('value')}' in {source_file}, but it is out of bounds: {value}")
                continue
            if metric == "sleep_score" and value == 0 and not _contains_explicit_zero_sleep_score(text):
                continue

            if value.is_integer():
                value = int(value)
            rows.append(
                observation(
                    date=date,
                    device=device,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source_file=source_file,
                    extraction_method=extraction_method,
                    confidence=confidence,
                    notes=notes,
                )
            )
            return

    if VERBOSE and metric == "avg_spo2_pct":
        print(f"DEBUG: Checking {source_file} for {metric}")
        print(f"DEBUG: No avg_spo2_pct match found. Text snippet:")
        # Print a snippet around oxygen/saturation/blood/SpO2/average
        snippet_keywords = ["oxygen", "saturation", "blood", "spo2", "average"]
        for line in text.splitlines():
            if any(k in line.lower() for k in snippet_keywords):
                print(f"  {line}")


def _within_metric_bounds(metric: str, value: float) -> bool:
    bounds = METRIC_BOUNDS.get(metric)
    if bounds is None:
        return True
    lower, upper = bounds
    return lower <= value <= upper


def parse_wellness_text(
    text: str,
    *,
    device: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
    page: int | None = None,
) -> list[dict[str, object]]:
    """Parse common non-CPAP sleep summary fields from extracted text."""

    date = infer_date(text, source_file, page=page)
    rows: list[dict[str, object]] = []
    specs = [
        ("total_sleep_minutes", ["(?:total\\s+)?sleep", "sleep\\s+duration", "time\\s+asleep", "asleep"], "minutes", "duration"),
        ("time_in_bed_minutes", ["time\\s+in\\s+bed", "in\\s+bed"], "minutes", "duration"),
        ("awake_minutes", ["awake", "wake"], "minutes", "duration"),
        ("rem_minutes", ["rem"], "minutes", "duration"),
        ("light_minutes", ["light", "core"], "minutes", "duration"),
        ("deep_minutes", ["deep"], "minutes", "duration"),
        ("sleep_score", ["sleep\\s+score", "\\bscore"], "score", "number"),
        ("sleep_efficiency_pct", ["sleep\\s+efficiency", "efficiency"], "pct", "number"),
        ("min_hr_bpm", ["lowest\\s+(?:heart\\s+rate|hr)", "min(?:imum)?\\s+(?:heart\\s+rate|hr)"], "bpm", "number"),
        ("avg_hr_bpm", ["average\\s+(?:heart\\s+rate|hr)", "avg\\.?\\s*(?:heart\\s+rate|hr)"], "bpm", "number"),
        ("avg_hrv_ms", ["average\\s+hrv", "avg\\.?\\s*hrv"], "ms", "number"),
        ("max_hrv_ms", ["max(?:imum)?\\s+hrv"], "ms", "number"),
        ("hrv_balance_score", ["hrv\\s+balance"], "score", "number"),
        ("avg_spo2_pct", ["average\\s+spo2", "avg\\.?\\s*spo2", "average\\s+oxygen", "avg\\.?\\s*oxygen", "oxygen\\s+saturation", "blood\\s+oxygen(?:\\s*\\W+)?\\s+average", "\\bspo2"], "pct", "number"),
        ("min_spo2_pct", ["lowest\\s+spo2", "min(?:imum)?\\s+spo2", "lowest\\s+oxygen", "min(?:imum)?\\s+oxygen"], "pct", "number"),
        ("respiratory_rate_bpm", ["respiratory\\s+rate", "respiration\\s+rate"], "breaths/min", "number"),
        ("temperature_deviation_c", ["temperature\\s+deviation", "temp\\.?\\s+deviation"], "C", "number"),
        ("readiness_score", ["readiness\\s+score", "\\breadiness"], "score", "number"),
    ]

    for metric, patterns, unit, kind in specs:
        if kind == "duration":
            add_duration_observation(
                rows,
                text,
                metric=metric,
                label_patterns=patterns,
                date=date,
                device=device,
                source_file=source_file,
                extraction_method=extraction_method,
                confidence=confidence,
                notes=notes,
            )
        else:
            add_number_observation(
                rows,
                text,
                metric=metric,
                unit=unit,
                label_patterns=patterns if metric != "avg_spo2_pct" else [
                    "average.*?spo2", 
                    "avg.*?spo2", 
                    "average.*?oxygen", 
                    "avg.*?oxygen", 
                    "oxygen.*?saturation", 
                    "blood.*?oxygen.*?average", 
                    "\\bspo2"
                ],
                date=date,
                device=device,
                source_file=source_file,
                extraction_method=extraction_method,
                confidence=confidence,
                notes=notes,
                page=page,
                flags=re.IGNORECASE | re.DOTALL if metric == "avg_spo2_pct" else re.IGNORECASE,
                value_pattern=r"\d{2,3}" if metric == "avg_spo2_pct" else r"[+-]?\d+(?:\.\d+)?",
            )

    breathing = re.search(
        r"\bbreathing(?:\s+(?:regularity|quality|label))?\s*[:\-]\s*(?P<label>[A-Za-z][A-Za-z ]{2,30})",
        text,
        re.IGNORECASE,
    )
    if breathing:
        rows.append(
            observation(
                date=date,
                device=device,
                metric="breathing_label",
                value=breathing.group("label").strip().splitlines()[0],
                unit="label",
                source_file=source_file,
                extraction_method=extraction_method,
                confidence="low" if confidence == "medium" else confidence,
                notes=notes,
            )
        )

    return rows


def parse_cpap_text(
    text: str,
    *,
    device: str,
    source_file: str,
    extraction_method: str,
    confidence: str,
    notes: str,
    page: int | None = None,
) -> list[dict[str, object]]:
    """Parse CPAP summary fields from OSCAR/SleepScope-style text."""

    date = infer_date(text, source_file, page=page)
    rows: list[dict[str, object]] = []
    add_duration_observation(
        rows,
        text,
        metric="cpap_usage_hours",
        label_patterns=["mask\\s+time", "usage", "cpap\\s+usage"],
        date=date,
        device=device,
        source_file=source_file,
        extraction_method=extraction_method,
        confidence=confidence,
        notes=notes,
        unit="hours",
        value_transform=lambda minutes: round(minutes / 60, 3),
    )
    number_specs = [
        ("cpap_ahi", ["\\bahi", "events\\s*/\\s*hour", "events\\s+per\\s+hour"], "events/hour"),
        ("cpap_cai", ["\\bcai", "clear\\s+airway\\s+index"], "events/hour"),
        ("cpap_oai", ["\\boai", "obstructive\\s+apnea\\s+index"], "events/hour"),
        ("cpap_pressure", ["95%?\\s+(?:pressure|press)", "pressure\\s+95%?", "\\bpressure"], "cmH2O"),
        ("cpap_leak_rate", ["mask\\s+leak", "(?:95%?\\s+)?leak(?:\\s+rate)?", "large\\s+leak"], "L/min"),
    ]
    for metric, patterns, unit in number_specs:
        add_number_observation(
            rows,
            text,
            metric=metric,
            unit=unit,
            label_patterns=patterns,
            date=date,
            device=device,
            source_file=source_file,
            extraction_method=extraction_method,
            confidence=confidence,
            notes=notes,
            page=page,
        )
    return rows


def infer_device_from_path(path: Path) -> str:
    """Infer a normalized device name from a sample folder or filename."""

    haystack = str(path).lower()
    if "oura4" in haystack or "ring 4" in haystack:
        return "Oura Ring 4 finger"
    if "oura3" in haystack or "ring 3" in haystack:
        return "Oura Ring 3 toe"
    if "samsung" in haystack or "sleepwatch" in haystack:
        return "Samsung Watch / SleepWatch"
    if "muse" in haystack:
        return "Muse"
    if "oscar" in haystack or "airsense" in haystack or "resmed" in haystack or "cpap" in haystack:
        return "ResMed AirSense 11"
    return "Unknown device"


def _date_from_match(match: re.Match[str]) -> str | None:
    values = match.groupdict()
    try:
        year = int(values.get("year") or datetime.now().year)
        if year < 100:
            year += 2000
        if values.get("month_name"):
            month_str = values["month_name"]
            # Simplified month name to number conversion if needed, or rely on strptime
            # datetime.strptime(f"{month_str} {values['day']} {year}", "%B %d %Y")
            # But the previous implementation used %b.
            # Let's keep it simple.
            return datetime.strptime(f"{month_str} {values['day']} {year}", "%B %d %Y").date().isoformat()
        return datetime(
            year,
            int(values["month"]),
            int(values["day"]),
        ).date().isoformat()
    except (KeyError, TypeError, ValueError):
        return None


def _join_notes(*parts: str) -> str:
    return "; ".join(part for part in parts if part)


def _contains_explicit_zero_sleep_score(text: str) -> bool:
    """Return True when the OCR text clearly shows a zero sleep score."""

    return bool(re.search(r"\bsleep\s+score\b\s*[:\-]?\s*0\b", text, flags=re.IGNORECASE))
