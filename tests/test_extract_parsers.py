from sleeppy.extract.common import parse_duration_to_minutes
from sleeppy.extract.oscar import parse_oscar_text
from sleeppy.extract.oura import parse_oura_text


def _metric_value(rows, metric):
    matches = [row for row in rows if row["metric"] == metric]
    assert matches, f"Expected metric {metric!r} in {rows}"
    return matches[0]["value"]


def test_parse_sleep_duration():
    rows = parse_oura_text(
        "Sleep 6h 52m",
        source_file="mock_oura.txt",
        device="Oura Ring 4 finger",
        extraction_method="ocr",
        confidence="medium",
    )
    assert _metric_value(rows, "total_sleep_minutes") == 412


def test_parse_average_hrv():
    rows = parse_oura_text(
        "Average HRV 24 ms",
        source_file="mock_oura.txt",
        device="Oura Ring 4 finger",
        extraction_method="ocr",
        confidence="medium",
    )
    assert _metric_value(rows, "avg_hrv_ms") == 24


def test_parse_ahi():
    rows = parse_oscar_text(
        "AHI 1.21",
        source_file="mock_oscar.txt",
        extraction_method="parsed_text",
        confidence="high",
    )
    assert _metric_value(rows, "cpap_ahi") == 1.21


def test_parse_mask_time_hms():
    rows = parse_oscar_text(
        "Mask Time: 1 hours, 39 minutes, 0 seconds",
        source_file="mock_oscar.txt",
        extraction_method="parsed_text",
        confidence="high",
    )
    assert _metric_value(rows, "cpap_usage_hours") == 1.65
    assert parse_duration_to_minutes("1 hours, 39 minutes, 0 seconds") == 99
