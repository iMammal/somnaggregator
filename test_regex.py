import re
from datetime import datetime

def _date_from_match(match):
    values = match.groupdict()
    year = int(values.get("year") or datetime.now().year)
    if year < 100:
        year += 2000
    month = int(values.get("month"))
    day = int(values.get("day"))
    return f"{year:04d}-{month:02d}-{day:02d}"

def infer_date(text, source_file):
    # OSCAR-style 2-digit year (MM-DD-YY)
    mm_dd_yy = re.search(r"\b(?P<month>\d{1,2})[-_/\.](?P<day>\d{1,2})[-_/\.](?P<year>\d{2})\b", text)
    if mm_dd_yy:
        return _date_from_match(mm_dd_yy)
    return None

print(f"Date: {infer_date('Sleep OSCAR 6-27-26 at 10.49.29 AM.pdf', '')}")
