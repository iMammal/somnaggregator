import re

# Test Oura
text_oura = "AVERAGE OXYGEN SATURATION 97%"
# Try being more permissive
pattern_oura = "oxygen.*?saturation"
regex = rf"{pattern_oura}\s*[:\-]?\s*(?P<value>[+-]?\d+(?:\.\d+)?)"

match = re.search(regex, text_oura, flags=re.IGNORECASE | re.DOTALL)
print(f"Oura match: {match}")
if match:
    print(f"Oura value: {match.group('value')}")

# Test Samsung
text_samsung = "Blood oxygen >\n= Average 94%"
# The OCR says "Blood oxygen >\n= Average 94%"
# My regex should match: "blood.*?oxygen.*?average"
pattern_samsung = "blood.*?oxygen.*?average"
regex2 = rf"{pattern_samsung}\s*[:\-]?\s*(?P<value>[+-]?\d+(?:\.\d+)?)"

match2 = re.search(regex2, text_samsung, flags=re.IGNORECASE | re.DOTALL)
print(f"Samsung match: {match2}")
if match2:
    print(f"Samsung value: {match2.group('value')}")
else:
    print("Samsung match failed")
