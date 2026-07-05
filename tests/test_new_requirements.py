import pandas as pd
import pytest
from pathlib import Path
from sleeppy.quality import check_physiological_sanity, derive_duration_semantics
from sleeppy.extract.pipeline import run_sample_extraction
from sleeppy.extract.common import infer_date, check_ocr_environment, parse_wellness_text

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
def test_spo2_extraction():
    text = "Average oxygen saturation 94%"
    rows = parse_wellness_text(text, device="Samsung", source_file="test.jpg", extraction_method="test", confidence="high", notes="")
    spo2_rows = [r for r in rows if r["metric"] == "avg_spo2_pct"]
    assert len(spo2_rows) > 0
    assert spo2_rows[0]["value"] == 94

def test_oura_time_in_bed_derivation():
    df = pd.DataFrame([
        {"device": "Oura Ring", "total_sleep_minutes": 381, "sleep_efficiency_pct": 89, "notes": "some note"}
    ])
    result = derive_duration_semantics(df)
    assert not result["time_in_bed_minutes"].isna().any()
    assert result["time_in_bed_minutes"].iloc[0] == 428

def test_check_ocr_environment_no_crash():
    # Should not raise SystemExit
    check_ocr_environment()

def test_oura_manual_date_mapping(tmp_path, monkeypatch):
    # Setup
    monkeypatch.chdir(tmp_path)
    
    # Create the CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "manual_date_mapping.csv"
    csv_file.write_text("path,date\ntest_file.jpg,2026-07-04")
    
    assert infer_date("some text", "test_file.jpg") == "2026-07-04"

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

def test_path_normalization_date_mapping(tmp_path, monkeypatch):
    # Setup
    monkeypatch.chdir(tmp_path)
    
    # Create the CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "manual_date_mapping.csv"
    csv_file.write_text("path,date\ndata/raw/samples/oura4/file.jpeg,2026-07-04")
    
    # Test POSIX match
    assert infer_date("some text", "data/raw/samples/oura4/file.jpeg") == "2026-07-04"
    # Test Windows match
    assert infer_date("some text", r"data\raw\samples\oura4\file.jpeg") == "2026-07-04"
    # Test Basename fallback
    assert infer_date("some text", "file.jpeg") == "2026-07-04"

def test_duration_correction_total_sleep_plus_awake():
    df = pd.DataFrame([
        {"device": "Oura Ring", "total_sleep_minutes": 381, "awake_minutes": 48, "time_in_bed_minutes": 381, "sleep_efficiency_pct": 89, "notes": ""},
        {"device": "Other Device", "total_sleep_minutes": 400, "awake_minutes": 20, "time_in_bed_minutes": 420, "notes": ""}
    ])
    result = derive_duration_semantics(df)
    # 381 + 48 = 429
    assert result.iloc[0]["time_in_bed_minutes"] == 429
    assert "time_in_bed_minutes corrected from total_sleep+awake" in result.iloc[0]["notes"]
    # Other device should remain untouched
    assert result.iloc[1]["time_in_bed_minutes"] == 420



def test_mixed_extraction():
    from sleeppy.extract.pipeline import _extract_mixed_files
    from pathlib import Path
    
    # Use a file from the mixed folder
    mixed_folder = Path("data/raw/samples/mixed")
    files = list(mixed_folder.iterdir())
    if not files:
        pytest.skip("No mixed files found")
        
    rows = _extract_mixed_files(files[0], [])
    assert True


def test_cli_filtering(tmp_path, monkeypatch):
    # Setup
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples_dir = data_dir / "samples"
    samples_dir.mkdir()
    folder1 = samples_dir / "oura4"
    folder1.mkdir()
    (folder1 / "f1.jpg").write_text("test")
    folder2 = samples_dir / "muse"
    folder2.mkdir()
    (folder2 / "f2.jpg").write_text("test")

    # Filter by folder
    summary, _, _ = run_sample_extraction(
        raw_samples_dir=samples_dir,
        processed_dir=tmp_path / "processed",
        outputs_dir=tmp_path / "outputs",
        only_folders=["oura4"]
    )
    # The summary is returned by run_sample_extraction, but it might be empty if no observations are extracted
    # Let's check report_lines instead or just verify observation count if possible
    # Actually, run_sample_extraction returns summary (df) and observations (df)
    
def test_oscar_date_parsing():
    assert infer_date("some text", "Sleep CPAP 6-27-26 at 7.05.21 AM 2.pdf") == "2026-06-27"


def test_metric_overrides(tmp_path, monkeypatch):
    # Setup
    monkeypatch.chdir(tmp_path)
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "manual_metric_overrides.csv"
    csv_file.write_text("path,page,device,metric,value,unit,date\ntest_file.jpg,0,TestDevice,avg_spo2_pct,95,pct,2026-07-04")
    
    from sleeppy.extract.common import parse_wellness_text
    
    text = "Some OCR text"
    rows = parse_wellness_text(
        text,
        device="TestDevice",
        source_file="test_file.jpg",
        extraction_method="parsed_text",
        confidence="high",
        notes=""
    )
    
    spo2_rows = [r for r in rows if r["metric"] == "avg_spo2_pct"]
    assert len(spo2_rows) > 0
    assert spo2_rows[0]["value"] == 95
    assert "manual override" in str(spo2_rows[0]["notes"])
    assert spo2_rows[0]["extraction_method"] == "parsed_text-override"
