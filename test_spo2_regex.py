import re

text_oura = "AVERAGE OXYGEN SATURATION 97%"
text_samsung = "Blood oxygen ... Average 94%"

# Patterns from common.py
patterns = ["oxygen\\s+saturation", "blood\\s+oxygen(?:\\s*\\W+)?\\s+average"]

for pattern in patterns:
    # Need to construct the full regex as add_number_observation would
    regex = rf"{pattern}\s*[:\-]?\s*(?P<value>[+-]?\d+(?:\.\d+)?)"
    print(f"Testing pattern: {pattern}")
    
    match_oura = re.search(regex, text_oura, flags=re.IGNORECASE)
    print(f"  Match Oura: {match_oura}, Group value: {match_oura.group('value') if match_oura else None}")
    
    match_samsung = re.search(regex, text_samsung, flags=re.IGNORECASE)
    print(f"  Match Samsung: {match_samsung}, Group value: {match_samsung.group('value') if match_samsung else None}")
