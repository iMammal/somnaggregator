# SleepPy

SleepPy is a personal sleep-analysis workspace for aligning and comparing nightly data from:

- Oura Ring 4 finger screenshots or exported summaries
- Oura Ring 3 toe screenshots or exported summaries
- Samsung Watch / SleepWatch data
- Muse sleep EEG screenshots or PDF summaries
- Muse S / MindMonitor sensor CSVs
- Oura API v2 cached JSON
- ResMed AirSense 11 CPAP data from OSCAR or SleepScope exports

This project is for exploratory wellness analysis only. It is not medical diagnosis, treatment advice, or a replacement for clinician review.

CPAP/OSCAR/SleepScope support is optional. If no CPAP files are present, extraction and notebook diagnostics should report that no CPAP metrics were detected without treating that as a failure.

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
      mixed/       # PDFs containing data from multiple devices
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
   - `data/raw/samples/mixed/` (For PDFs containing multiple devices)

   Screenshots can also be grouped by wake/report date to avoid manual date rows for generic phone filenames:

   ```text
   data/raw/samples/oura4/2026-07-09/IMG_1087.PNG
   data/raw/samples/oura3/2026-07-09/IMG_1088.PNG
   data/raw/samples/samsung_watch/2026-07-09/Screenshot.png
   data/raw/samples/muse/2026-07-09/Screenshot.png
   data/raw/samples/mind_monitor/2026-07-09/raw/museMonitor_....csv
   ```

   Manual `data/manual_date_mapping.csv` rows still take priority. Otherwise the extractor uses an explicit filename date first, then a parent folder named `YYYY-MM-DD`, then any supported OCR/content date.

   Real screenshots, PDFs, exports, and reports contain personal health data and should stay out of git. The repository ignores raw data by default and only preserves sample directory placeholders with `.gitkeep`.
3. Run first-pass extraction:

   ```powershell
   py -m sleeppy.extract
   ```

4. Open `sample.ipynb` to load the processed CSVs, inspect nightly tables, and plot sleep duration, HRV, SpO2, and CPAP AHI over time.

The extractor also scans image/PDF files directly under `data/raw/` as a convenience for the current project layout. Use `py -m sleeppy.extract --no-legacy-raw` to scan only `data/raw/samples/`.

## Oura API Ingestion

Oura API support is additive; the screenshot parser remains available. API JSON contains personal health data and should not be committed.

1. Create a personal Oura API token in your Oura account.
2. Set it in your shell or a local `.env` file:

   ```powershell
   $env:OURA_TOKEN = "your-token"
   ```

3. Fetch a date range into the raw API cache:

   ```powershell
   py -m sleeppy.api.oura_fetch --start-date 2026-07-09 --end-date 2026-07-09
   ```

   This writes cached JSON under `data/raw/api/oura/`, which is ignored by git.

4. Run extraction from cached API data:

   ```powershell
   py -m sleeppy.extract --no-legacy-raw --include-oura-api
   ```

   To process only cached Oura API files:

   ```powershell
   py -m sleeppy.extract --no-legacy-raw --only-folder oura_api
   ```

5. In `sample.ipynb`, set `RUN_EXTRACTION=True` after fetching new API JSON. The notebook can then compare `Oura API` rows with Oura screenshot rows on overlapping dates and metrics.

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

- `data/processed/device_observations_long.csv`: one row per extracted metric value with `night_date`, `device`, `metric`, `value`, `unit`, `source_file`, `extraction_method`, `confidence`, and `notes`.
- `data/processed/nightly_summary.csv`: one row per date/device with the best available value for each summary metric.
- `outputs/extraction_report.md`: extraction counts, confidence summary, skipped-file notes, and OCR setup notes.

The extractor prefers PDF parsed text with PyMuPDF, then OCR with pytesseract, then a manual-fallback note. It does not try to digitize line graphs. It only extracts summary metrics and simple CPAP summary text at this stage.

If no CPAP/OSCAR/SleepScope files are present, `cpap_ahi`, `cpap_usage_hours`, `cpap_leak_rate`, and `cpap_pressure` remain empty. This is expected for non-CPAP datasets.

On Windows, `pytesseract` still needs the separate Tesseract executable. Install it and make sure `tesseract.exe` is on PATH, or set `pytesseract.pytesseract.tesseract_cmd` in a notebook cell.

Check the active notebook/kernel OCR setup with:

```python
from sleeppy.extract import check_ocr_environment
check_ocr_environment()
```

If `image_ocr_ready` is `False`, image screenshots will not extract values. PDF files with selectable text can still work when `pymupdf_installed` is `True`.

## Canonical Metrics And Plotting Expectations

Extraction normalizes raw OCR labels into stable canonical metric names before writing CSV files. Examples:

- `total sleep`, `sleep duration`, `asleep`, and `time asleep` -> `total_sleep_minutes`
- `average HRV` and `avg HRV` -> `avg_hrv_ms`
- `HRV balance` -> `hrv_balance_score`
- `oxygen saturation`, `SpO2`, `average oxygen`, and `avg oxygen` -> `avg_spo2_pct`
- `AHI`, `events/hour`, and `events per hour` -> `cpap_ahi`
- `usage`, `CPAP usage`, and `mask time` -> `cpap_usage_hours`
- `mask leak`, `leak`, and `leak rate` -> `cpap_leak_rate`

The notebook trend plots look for these canonical plotting metrics:

- `total_sleep_minutes`
- `avg_hrv_ms`
- `avg_spo2_pct`
- `cpap_ahi` when CPAP data is present

Metrics can be extracted but still not plottable as trends if the source screenshot/PDF does not expose a parseable `night_date`. In that case the value remains in `device_observations_long.csv` and is grouped as `undated` in `nightly_summary.csv`.

Use this diagnostic helper after extraction:

```python
from sleeppy.quality import describe_extraction_outputs

diagnostics = describe_extraction_outputs(nightly_summary, observations)
```

It reports row counts, devices, detected metric names, canonical metrics available, plotting metrics missing, and source files that contributed values.

## Git Hygiene And Fixtures

The root `.gitignore` excludes:

- `.idea/`
- Python caches, pytest cache, notebook checkpoints, and virtualenvs
- generated `data/interim/`, `data/processed/`, and `outputs/`
- real raw health data under `data/raw/`

Only `.gitkeep` placeholders are intended to keep empty raw sample directories visible. Synthetic or de-identified test fixtures under `tests/fixtures/` are safe to track.

If IDE files were already staged or tracked before `.gitignore` was added, untrack them without deleting your local PyCharm settings:

```powershell
git rm -r --cached .idea
```

## Planned Import Pipeline

Later importers can add true CSV/JSON ingestion for Oura, Samsung Health, Muse, OSCAR, and SleepScope. They should emit the same long-form observation schema and then write cleaned tables to `data/interim/` or `data/processed/`.

## Minimal Example

```python
from sleeppy.extract import run_sample_extraction

nightly_summary, observations, report_path = run_sample_extraction()
```
