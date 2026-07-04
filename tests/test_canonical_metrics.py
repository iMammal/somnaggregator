from pathlib import Path

import pandas as pd

from sleeppy.quality import describe_extraction_outputs, observations_to_nightly_summary
from sleeppy.schema import PLOT_METRICS, ensure_observations_frame, normalize_metric_name


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_raw_metric_aliases_normalize_to_canonical_names():
    aliases = {
        "total sleep": "total_sleep_minutes",
        "sleep duration": "total_sleep_minutes",
        "asleep": "total_sleep_minutes",
        "average HRV": "avg_hrv_ms",
        "HRV balance": "hrv_balance_score",
        "oxygen saturation": "avg_spo2_pct",
        "SpO2": "avg_spo2_pct",
        "avg oxygen": "avg_spo2_pct",
        "AHI": "cpap_ahi",
        "events/hour": "cpap_ahi",
        "usage": "cpap_usage_hours",
        "mask leak": "cpap_leak_rate",
    }

    for raw_label, expected in aliases.items():
        assert normalize_metric_name(raw_label) == expected


def test_observations_normalize_legacy_columns_and_metrics():
    observations = pd.DataFrame(
        [
            {
                "date": "2026-06-08",
                "device": "Oura Ring 4 finger",
                "metric": "avg_hrv",
                "value": 24,
                "unit": "ms",
                "source_file": "mock.png",
                "extraction_method": "ocr",
                "confidence": "medium",
                "notes": "",
            },
            {
                "date": "2026-06-08",
                "device": "ResMed AirSense 11",
                "metric": "cpap_mask_minutes",
                "value": 99,
                "unit": "minutes",
                "source_file": "mock.pdf",
                "extraction_method": "parsed_text",
                "confidence": "high",
                "notes": "",
            },
        ]
    )

    normalized = ensure_observations_frame(observations)
    assert list(normalized["metric"]) == ["avg_hrv_ms", "cpap_usage_hours"]
    assert normalized.loc[1, "value"] == 1.65
    assert "night_date" in normalized.columns


def test_nightly_summary_gets_expected_canonical_columns():
    observations = pd.DataFrame(
        [
            {
                "night_date": "2026-06-08",
                "device": "Oura Ring 4 finger",
                "metric": "sleep duration",
                "value": 412,
                "unit": "minutes",
                "source_file": "mock.png",
                "extraction_method": "ocr",
                "confidence": "medium",
                "notes": "",
            },
            {
                "night_date": "2026-06-08",
                "device": "Oura Ring 4 finger",
                "metric": "Average HRV",
                "value": 24,
                "unit": "ms",
                "source_file": "mock.png",
                "extraction_method": "ocr",
                "confidence": "medium",
                "notes": "",
            },
        ]
    )

    summary = observations_to_nightly_summary(observations)
    assert "total_sleep_minutes" in summary.columns
    assert "avg_hrv_ms" in summary.columns
    assert summary.loc[0, "total_sleep_minutes"] == 412
    assert summary.loc[0, "avg_hrv_ms"] == 24


def test_plot_metrics_are_discoverable_and_missing_metrics_are_reported():
    observations = pd.DataFrame(
        [
            {
                "night_date": "2026-06-08",
                "device": "Oura Ring 4 finger",
                "metric": "total_sleep_minutes",
                "value": 412,
                "unit": "minutes",
                "source_file": "mock.png",
                "extraction_method": "ocr",
                "confidence": "medium",
                "notes": "",
            }
        ]
    )
    summary = observations_to_nightly_summary(observations)
    diagnostics = describe_extraction_outputs(summary, observations, print_output=False)

    assert "total_sleep_minutes" in diagnostics["canonical_metrics_available"]
    assert "total_sleep_minutes" not in diagnostics["expected_plot_metrics_missing"]
    assert "cpap_ahi" not in diagnostics["expected_plot_metrics_missing"]
    assert "cpap_ahi" in diagnostics["optional_plot_metrics_missing"]
    assert diagnostics["cpap_detected"] is False
    for metric in PLOT_METRICS:
        if metric not in {"total_sleep_minutes", "cpap_ahi"}:
            assert metric in diagnostics["expected_plot_metrics_missing"]


def test_no_cpap_data_present_is_not_a_failure():
    observations = pd.DataFrame(
        [
            {
                "night_date": "2026-06-08",
                "device": "Oura Ring 4 finger",
                "metric": "avg_hrv_ms",
                "value": 24,
                "unit": "ms",
                "source_file": "mock.png",
                "extraction_method": "ocr",
                "confidence": "medium",
                "notes": "",
            }
        ]
    )
    summary = observations_to_nightly_summary(observations)
    diagnostics = describe_extraction_outputs(summary, observations, print_output=False)

    assert diagnostics["cpap_detected"] is False
    assert diagnostics["cpap_metrics_detected"] == []
    assert "cpap_ahi" not in diagnostics["expected_plot_metrics_missing"]
    assert "cpap_ahi" in diagnostics["optional_plot_metrics_missing"]


def test_synthetic_cpap_metrics_flow_to_nightly_summary():
    observations = pd.read_csv(FIXTURES_DIR / "synthetic_cpap_observations.csv")
    normalized = ensure_observations_frame(observations)
    summary = observations_to_nightly_summary(normalized)
    diagnostics = describe_extraction_outputs(summary, normalized, print_output=False)

    assert summary.loc[0, "cpap_ahi"] == 1.2
    assert summary.loc[0, "cpap_usage_hours"] == 7.5
    assert summary.loc[0, "cpap_leak_rate"] == 3.4
    assert summary.loc[0, "cpap_pressure"] == 9.8
    assert diagnostics["cpap_detected"] is True
    assert "cpap_ahi" in diagnostics["canonical_metrics_available"]
    assert "cpap_ahi" not in diagnostics["optional_plot_metrics_missing"]


def test_gitignore_privacy_patterns_are_present():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    required_patterns = [
        ".idea/",
        "__pycache__/",
        ".pytest_cache/",
        ".ipynb_checkpoints/",
        ".venv/",
        "data/raw/*",
        "!data/raw/.gitkeep",
        "!data/raw/samples/**/.gitkeep",
        "data/interim/*",
        "data/processed/*",
        "outputs/*",
        "!tests/fixtures/**",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore
