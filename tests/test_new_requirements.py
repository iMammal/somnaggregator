import pandas as pd
import pytest
from pathlib import Path
from sleeppy.quality import check_physiological_sanity
from sleeppy.extract.pipeline import run_sample_extraction
from sleeppy.extract.common import infer_date

def test_check_physiological_sanity():
    df = pd.DataFrame([
        {"night_date": "2026-07-04", "device": "TestDevice", "avg_spo2_pct": 50, "avg_hr_bpm": 20, "min_hr_bpm": 10, "sleep_efficiency_pct": 110, "total_sleep_minutes": 100, "time_in_bed_minutes": 20, "rem_minutes": 10, "light_minutes": 10, "deep_minutes": 10, "awake_minutes": 10}
    ])
    warnings = check_physiological_sanity(df)
    assert len(warnings) > 0
    assert any("avg_spo2_pct=50" in w for w in warnings)
    assert any("avg_hr_bpm=20" in w for w in warnings)
    assert any("min_hr_bpm=10" in w for w in warnings)
    assert any("sleep_efficiency_pct=110" in w for w in warnings)
    assert any("total_sleep_minutes=100" in w for w in warnings)
    assert any("time_in_bed_minutes=20 < total_sleep_minutes=100" in w for w in warnings)
    assert any("stage sum=30 != total_sleep_minutes=100" in w for w in warnings)
    assert any("stage sum=40 != time_in_bed_minutes=20" in w for w in warnings)

def test_date_extraction():
    # Manual mapping
    assert infer_date("some text", "Screenshot_20260608-095716_Muse.jpg") == "2026-06-08"
    
    # Regex
    assert infer_date("2026-06-08", "test.jpg") == "2026-06-08"
    assert infer_date("May 28, 2026", "test.jpg") == "2026-05-28"

def test_relative_path_redaction(tmp_path, monkeypatch):
    # Setup
    monkeypatch.chdir(tmp_path)
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples_dir = data_dir / "samples"
    samples_dir.mkdir()
    muse_dir = samples_dir / "muse"
    muse_dir.mkdir()
    (muse_dir / "test.jpg").write_text("dummy")

    # Run extraction
    _, _, report_path = run_sample_extraction(
        raw_samples_dir=samples_dir,
        processed_dir=tmp_path / "processed",
        outputs_dir=tmp_path / "outputs",
        verbose=False
    )
    
    report_content = report_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in report_content
    assert "data/samples/muse" in report_content or "data\\samples\\muse" in report_content

def test_optional_cpap_absence(tmp_path):
    # Setup
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples_dir = data_dir / "samples"
    samples_dir.mkdir()
    # No oscar folder
    
    # Run extraction
    _, _, report_path = run_sample_extraction(
        raw_samples_dir=samples_dir,
        processed_dir=tmp_path / "processed",
        outputs_dir=tmp_path / "outputs",
        verbose=False
    )
    
    report_content = report_path.read_text(encoding="utf-8")
    assert "No CPAP metrics detected; CPAP/OSCAR/SleepScope is optional." in report_content
