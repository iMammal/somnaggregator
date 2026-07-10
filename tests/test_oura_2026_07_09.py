from sleeppy.extract.oura import parse_oura_text

def _metric_value(rows, metric):
    matches = [row for row in rows if row["metric"] == metric]
    assert matches, f"Expected metric {metric!r} in {rows}"
    return matches[0]["value"]

def test_oura_2026_07_09_deep_sleep_parsing():
    # OCR-like text simulating "deep 1h 05m" or variations
    text = "deep 1h 05m"
    rows = parse_oura_text(
        text,
        source_file="mock_oura.txt",
        device="Oura Ring 4 finger",
        extraction_method="ocr",
        confidence="medium",
    )
    assert _metric_value(rows, "deep_minutes") == 65

def test_oura_2026_07_09_explicit_tib_precedence():
    # Simulate both explicit TIB (7h36m = 456m) and derived (399m sleep + 57m awake = 456m)
    # The requirement is to prefer the explicit one if it's there.
    # If I provide both, TIB from OCR and total_sleep + awake, it should prefer the explicit one.
    
    # Actually, in Oura parsing, total_sleep and awake are usually parsed from separate lines.
    # Let's see if we can provide a text that includes both.
    text = "total sleep 6h39m\ntime in bed 7h36m\nawake 0h57m"
    
    rows = parse_oura_text(
        text,
        source_file="mock_oura.txt",
        device="Oura Ring 4 finger",
        extraction_method="ocr",
        confidence="medium",
    )
    
    # TIB should be 456 (7h36m)
    assert _metric_value(rows, "time_in_bed_minutes") == 456
    # It should not have been overwritten by derived (399 + 57 = 456), which is the same value.
    # What if they were different? 
    # Say TIB is 460. Derived is 456.
    
    text_different = "total sleep 6h39m\ntime in bed 7h40m\nawake 0h57m"
    rows = parse_oura_text(
        text_different,
        source_file="mock_oura.txt",
        device="Oura Ring 4 finger",
        extraction_method="ocr",
        confidence="medium",
    )
    
    # It should prefer the explicit 7h40m = 460m.
    assert _metric_value(rows, "time_in_bed_minutes") == 460
