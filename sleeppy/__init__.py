"""Readable pandas helpers for personal sleep-data exploration."""

from .compare import metric_pivot, stage_disagreement_matrix, stage_pivot, summarize_by_device_night
from .extract import check_ocr_environment, run_sample_extraction
from .loaders import load_manual_summaries, parse_sleep_window, summaries_to_timelines
from .quality import confidence_by_device, missingness_by_device, observations_to_nightly_summary
from .resample import resample_timelines
from .schema import SleepTimeline
from .timeline import align_timelines, make_timeline
from .viz import (
    plot_cpap_panel,
    plot_hr_hrv_overlay,
    plot_sleep_stage_timeline,
    plot_stage_disagreement_heatmap,
)

__all__ = [
    "SleepTimeline",
    "align_timelines",
    "check_ocr_environment",
    "load_manual_summaries",
    "make_timeline",
    "metric_pivot",
    "parse_sleep_window",
    "plot_cpap_panel",
    "plot_hr_hrv_overlay",
    "plot_sleep_stage_timeline",
    "plot_stage_disagreement_heatmap",
    "confidence_by_device",
    "missingness_by_device",
    "observations_to_nightly_summary",
    "resample_timelines",
    "run_sample_extraction",
    "stage_disagreement_matrix",
    "stage_pivot",
    "summaries_to_timelines",
    "summarize_by_device_night",
]
