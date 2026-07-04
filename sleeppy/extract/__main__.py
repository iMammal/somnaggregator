"""CLI entry point for `python -m sleeppy.extract`."""

from __future__ import annotations

import argparse

from .pipeline import run_sample_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first-pass sleep summary metrics from sample files.")
    parser.add_argument("--raw-samples-dir", default="data/raw/samples")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--no-legacy-raw", action="store_true", help="Do not also scan files directly under data/raw.")
    args = parser.parse_args()

    summary, observations, report_path = run_sample_extraction(
        raw_samples_dir=args.raw_samples_dir,
        processed_dir=args.processed_dir,
        outputs_dir=args.outputs_dir,
        include_legacy_raw=not args.no_legacy_raw,
    )
    print(f"Wrote {len(summary)} nightly rows and {len(observations)} observations.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
