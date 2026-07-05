import re

text = "Average oxygen saturation 94%"
pattern = r"average\s+oxygen\s+saturation\s*[:\-]?\s*(?P<value>[+-]?\d+(?:\.\d+)?)"
match = re.search(pattern, text, flags=re.IGNORECASE)
print(f"Match: {match}")
if match:
    print(f"Value: {match.group('value')}")
