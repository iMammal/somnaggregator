import re

def parse_duration_to_minutes(text: str) -> int | None:
    value = text.strip().lower()
    hms = re.search(
        r"(?:(?P<hours>\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h))?\s*,?\s*"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m))?\s*,?\s*"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s))?",
        value,
    )
    if not hms:
        return None
    print(f"Hours: {hms.group('hours')}, Minutes: {hms.group('minutes')}, Seconds: {hms.group('seconds')}")

    hours = float(hms.group('hours') or 0)
    minutes = float(hms.group('minutes') or 0)
    seconds = float(hms.group('seconds') or 0)
    if hours == 0 and minutes == 0 and seconds == 0:
        return None
    return int(round(hours * 60 + minutes + seconds / 60))

text = "Light 5h 09m"
pattern = "light"
regex = rf"{pattern}\s*[:\-]?\s*(?P<duration>\d+(?:\.\d+)?\s*(?:hours?|hrs?|h)(?:\s*,?\s*\d+(?:\.\d+)?\s*(?:minutes?|mins?|m))?(?:\s*,?\s*\d+(?:\.\d+)?\s*(?:seconds?|secs?|s))?|\d+(?:\.\d+)?\s*(?:minutes?|mins?|m))"

match = re.search(regex, text, flags=re.IGNORECASE)
print(f"Match (5h 09m): {match}")
if match:
    duration = match.group('duration')
    print(f"Group duration (5h 09m): {duration}")
    print(f"Result (5h 09m): {parse_duration_to_minutes(duration)}")

text2 = "Light 5h09m"
match2 = re.search(regex, text2, flags=re.IGNORECASE)
print(f"Match (5h09m): {match2}")
if match2:
    duration2 = match2.group('duration')
    print(f"Group duration (5h09m): {duration2}")
    print(f"Result (5h09m): {parse_duration_to_minutes(duration2)}")
