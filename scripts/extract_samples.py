"""Run first-pass extraction from data/raw/samples into processed CSVs."""

from sleeppy.extract import run_sample_extraction


if __name__ == "__main__":
    summary, observations, report_path = run_sample_extraction()
    print(f"Wrote {len(summary)} nightly rows and {len(observations)} observations.")
    print(f"Report: {report_path}")
