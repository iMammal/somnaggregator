import hashlib
from pathlib import Path

def _get_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

path = Path("data/raw/samples/mixed/Sleep All 6-28-26 at 9.27.49 AM.pdf")
print(f"Hash: {_get_file_hash(path)}")
