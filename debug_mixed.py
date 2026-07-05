from pathlib import Path
from sleeppy.extract.pipeline import _extract_mixed_files
from sleeppy.extract.common import load_date_mapping, source_file_label, extract_pdf_pages_text

path = Path("data/raw/samples/mixed/Sleep All 6-28-26 at 9.27.49 AM.pdf")
print("Path: [redacted]")
print(f"Text pages: {len(extract_pdf_pages_text(path))}")
for i, text in enumerate(extract_pdf_pages_text(path)):
    print(f"Page {i+1} text length: {len(text)}")
    print(f"Page {i+1} preview: {text[:100]}")

rows = _extract_mixed_files(path)
print(f"Extracted rows: {len(rows)}")
for r in rows:
    print(r)
