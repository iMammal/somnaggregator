# SleepPy

SleepPy is a personal sleep-analysis workspace for aligning and comparing nightly data from:

- Oura Ring 4 finger screenshots or exported summaries
- Oura Ring 3 toe screenshots or exported summaries
- Samsung Watch / SleepWatch data
- Muse sleep EEG screenshots or PDF summaries
- ResMed AirSense 11 CPAP data from OSCAR or SleepScope exports

This project is for exploratory wellness analysis only. It is not medical diagnosis, treatment advice, or a replacement for clinician review.

## Project Structure

```text
data/
  raw/         # Original screenshots, PDFs, CSVs, JSON exports
    samples/
      oura4/
      oura3/
      samsung_watch/
      muse/
      oscar/
  interim/     # Manual-entry CSVs and lightly cleaned extracts
  processed/   # Extracted normalized CSVs and aligned/resampled analysis tables
notebooks/     # Follow-up notebooks
sleeppy/       # Reusable pandas + matplotlib helpers
outputs/       # Figures and exported reports
sample.ipynb   # Main extraction and analysis notebook
```

## Current Workflow

1. Install the runtime dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

   If PyCharm/Jupyter is using a different interpreter, install into that exact kernel instead:

   ```python
   import sys
   !"{sys.executable}" -m pip install -r requirements.txt
   ```

2. Put 1-2 representative screenshots or PDFs per device in:
   - `data/raw/samples/oura4/`
   - `data/raw/samples/oura3/`
   - `data/raw/samples/samsung_watch/`
   - `data/raw/samples/muse/`
   - `data/raw/samples/oscar/`
3. Run first-pass extraction:

   ```powershell
   py -m sleeppy.extract
   ```

4. Open `sample.ipynb` to load the processed CSVs, inspect nightly tables, and plot sleep duration, HRV, SpO2, and CPAP AHI over time.

The extractor also scans image/PDF files directly under `data/raw/` as a convenience for the current project layout. Use `py -m sleeppy.extract --no-legacy-raw` to scan only `data/raw/samples/`.

## Helper Modules

- `sleeppy.loaders`: loads manually entered summaries and converts them to coarse timeline segments.
- `sleeppy.schema`: defines normalized sleep-stage labels and the `SleepTimeline` container.
- `sleeppy.timeline`: builds and aligns device timelines on clock time.
- `sleeppy.resample`: resamples timelines into fixed 15-minute or 30-minute bins.
- `sleeppy.compare`: creates stage pivots, disagreement matrices, metric pivots, and night summaries.
- `sleeppy.extract`: first-pass screenshot/PDF extraction for Oura, Samsung/SleepWatch, Muse, and OSCAR.
- `sleeppy.quality`: provenance, confidence, missingness, and report helpers.
- `sleeppy.viz`: plots sleep-stage timelines, HR/HRV overlays, CPAP panels, and disagreement heatmaps.

## Extraction Pipeline

The first-pass extractor creates two normalized files:

- `data/processed/device_observations_long.csv`: one row per extracted metric value with `source_file`, `extraction_method`, `confidence`, and `notes`.
- `data/processed/nightly_summary.csv`: one row per date/device with the best available value for each summary metric.
- `outputs/extraction_report.md`: extraction counts, confidence summary, skipped-file notes, and OCR setup notes.

The extractor prefers PDF parsed text with PyMuPDF, then OCR with pytesseract, then a manual-fallback note. It does not try to digitize line graphs. It only extracts summary metrics and simple CPAP summary text at this stage.

On Windows, `pytesseract` still needs the separate Tesseract executable. Install it and make sure `tesseract.exe` is on PATH, or set `pytesseract.pytesseract.tesseract_cmd` in a notebook cell.

Check the active notebook/kernel OCR setup with:

```python
from sleeppy.extract import check_ocr_environment
check_ocr_environment()
```

If `image_ocr_ready` is `False`, image screenshots will not extract values. PDF files with selectable text can still work when `pymupdf_installed` is `True`.

## Planned Import Pipeline

Later importers can add true CSV/JSON ingestion for Oura, Samsung Health, Muse, OSCAR, and SleepScope. They should emit the same long-form observation schema and then write cleaned tables to `data/interim/` or `data/processed/`.

## Minimal Example

```python
from sleeppy.extract import run_sample_extraction

nightly_summary, observations, report_path = run_sample_extraction()
```
