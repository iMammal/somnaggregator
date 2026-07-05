import fitz
import time
from pathlib import Path

path = Path("data/raw/samples/mixed/Sleep All 6-28-26 at 9.27.49 AM.pdf")
start = time.time()
with fitz.open(path) as doc:
    print(f"Opened {path.name} in {time.time() - start:.2f}s")
    for page in doc:
        _ = page.get_text("text")
print(f"Processed {path.name} in {time.time() - start:.2f}s")
