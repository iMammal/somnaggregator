"""Helpers for building and aligning sleep timelines."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from .loaders import parse_sleep_window
from .schema import SleepTimeline, concat_timeline_frames, ensure_timeline_frame


def make_timeline(
    device: str,
    night_date: str,
    segments: Iterable[Mapping[str, object]] | pd.DataFrame,
    source: str | None = None,
    notes: str | None = None,
) -> SleepTimeline:
    """Build a SleepTimeline from segment dictionaries or a dataframe.

    Segment ``start`` and ``end`` values can be full datetimes or clock strings
    such as ``"23:10"`` and ``"06:45"``.
    """

    frame = segments.copy() if isinstance(segments, pd.DataFrame) else pd.DataFrame(list(segments))
    if frame.empty:
        return SleepTimeline(device=device, night_date=night_date, segments=frame, source=source, notes=notes)

    for row_index, row in frame.iterrows():
        start, end = parse_sleep_window(night_date, row["start"], row["end"])
        frame.at[row_index, "start"] = start
        frame.at[row_index, "end"] = end

    return SleepTimeline(device=device, night_date=night_date, segments=frame, source=source, notes=notes)


def align_timelines(timelines: Iterable[SleepTimeline | pd.DataFrame]) -> pd.DataFrame:
    """Align many device timelines on local clock time."""

    frames = []
    for timeline in timelines:
        if isinstance(timeline, SleepTimeline):
            if timeline.segments.empty:
                continue
            frames.append(timeline.to_frame())
        else:
            frames.append(ensure_timeline_frame(timeline))

    aligned = concat_timeline_frames(frames)
    if aligned.empty:
        return aligned

    aligned["clock_start"] = aligned["start"]
    aligned["clock_end"] = aligned["end"]
    aligned["start_clock"] = aligned["start"].dt.strftime("%H:%M")
    aligned["end_clock"] = aligned["end"].dt.strftime("%H:%M")
    return aligned.sort_values(["night_date", "start", "device"]).reset_index(drop=True)
