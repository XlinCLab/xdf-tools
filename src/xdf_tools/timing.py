"""Sample-timing quality checks: gaps (dropped samples), jitter, and effective vs. nominal sampling rate,
with a verdict per stream, an overall pass/fail per file, and a text report."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xdf_tools.stream import XDFStream

# A stream with a verdict starting with one of these has failed validation
FAIL_PREFIXES = ("SEVERE", "CORRUPT", "NO DATA", "FAILED")

# An inter-sample interval must exceed this multiple of the nominal period to count as a
# discrete gap (dropped sample(s)) rather than normal jitter.
GAP_OUTLIER_FACTOR = 1.5
# Sample-to-sample interval std (excluding discrete gaps), as a percentage of the nominal
# period, above which timing is flagged as jittery even though no gaps were detected.
JITTER_THRESHOLD_PCT = 15.0
# A stream shorter than this fraction of the longest stream in its file is considered cut off.
MIN_DURATION_FRACTION = 0.5
# Maximum relative deviation of effective from nominal sampling rate, with no discrete gaps.
RATE_MISMATCH_TOLERANCE = 0.02
# A stream with a smaller fraction of missing samples than this is "ok" despite having gaps.
NEGLIGIBLE_MISSING_FRACTION = 0.005
# Streams with at most this many gaps and a smaller missing fraction than SALVAGEABLE_MISSING_FRACTION are salvageable.
SALVAGEABLE_MAX_GAPS = 3
SALVAGEABLE_MISSING_FRACTION = 0.15


def _duration(time_stamps: np.ndarray) -> float:
    if len(time_stamps) < 2:
        return 0.0
    return float(time_stamps[-1] - time_stamps[0])


def summarize_gaps(time_stamps: np.ndarray, nominal_srate: float) -> dict:
    """Summarize sample-timing quality: gap count/size, effective vs. nominal rate, and
    jitter (sample-to-sample interval variability among non-gap intervals)."""
    time_stamps = np.asarray(time_stamps, dtype=np.float64)
    n_samples = len(time_stamps)
    duration = _duration(time_stamps)

    if n_samples < 2 or nominal_srate <= 0:
        return {
            "n_samples": n_samples,
            "duration": duration,
            "effective_srate": 0.0,
            "nominal_srate": nominal_srate,
            "n_gaps": 0,
            "total_missing": 0,
            "max_gap_missing": 0,
            "jitter_std": 0.0,
            "jitter_pct": 0.0,
        }

    effective_srate = (n_samples - 1) / duration if duration > 0 else 0.0
    nominal_period = 1.0 / nominal_srate

    intervals = np.abs(np.diff(time_stamps))
    outlier_mask = intervals > (GAP_OUTLIER_FACTOR * nominal_period)
    outlier_gaps = intervals[outlier_mask]
    implied_missing = np.round(outlier_gaps / nominal_period).astype(int) - 1
    implied_missing = implied_missing[implied_missing > 0]

    regular_intervals = intervals[~outlier_mask]
    jitter_std = float(regular_intervals.std()) if len(regular_intervals) > 0 else 0.0
    jitter_pct = jitter_std / nominal_period * 100

    return {
        "n_samples": n_samples,
        "duration": duration,
        "effective_srate": effective_srate,
        "nominal_srate": nominal_srate,
        "n_gaps": len(implied_missing),
        "total_missing": int(implied_missing.sum()) if len(implied_missing) else 0,
        "max_gap_missing": int(implied_missing.max()) if len(implied_missing) else 0,
        "jitter_std": jitter_std,
        "jitter_pct": jitter_pct,
    }


def locate_gaps(time_stamps: np.ndarray, nominal_srate: float) -> list[dict]:
    """Return the time range and size of each detected gap (dropped-sample interval),
    in seconds relative to the first timestamp. Empty for irregular streams
    or streams without any detected gaps."""
    time_stamps = np.asarray(time_stamps, dtype=np.float64)
    if len(time_stamps) < 2 or nominal_srate <= 0:
        return []

    nominal_period = 1.0 / nominal_srate
    t0 = time_stamps[0]
    intervals = np.abs(np.diff(time_stamps))
    outlier_mask = intervals > (GAP_OUTLIER_FACTOR * nominal_period)

    gaps = []
    for idx in np.flatnonzero(outlier_mask):
        n_missing = int(round(intervals[idx] / nominal_period)) - 1
        if n_missing <= 0:
            continue
        gaps.append({
            "start": float(time_stamps[idx] - t0),
            "end": float(time_stamps[idx + 1] - t0),
            "n_missing": n_missing,
        })
    return gaps


def describe_gap_distribution(gaps: list[dict], duration: float, gap_concentration_threshold: float = 0.7) -> str:
    """Describe where detected gaps fall within the recording: whether the missing
    samples are concentrated within one portion of the stream or spread throughout.
    Distinguishes a stream that's fine except for one bad stretch from one with
    pervasive dropouts, which have the same gap count/fraction but very different
    implications for salvageability.

    gap_concentration_threshold (float): Fraction of a stream's missing samples
        that must fall within one third of the recording for its gaps to be
        described as "concentrated" rather than "spread throughout".
    """
    if not gaps or duration <= 0:
        return "n/a"

    thirds = [0, 0, 0]
    labels = ("first third", "middle third", "last third")
    for gap in gaps:
        midpoint = (gap["start"] + gap["end"]) / 2
        idx = min(int((midpoint / duration) * 3), 2)
        thirds[idx] += gap["n_missing"]

    total_missing = sum(thirds)
    if total_missing == 0:
        return "n/a"
    dominant_idx = max(range(3), key=lambda i: thirds[i])
    if thirds[dominant_idx] / total_missing >= gap_concentration_threshold:
        return f"concentrated in {labels[dominant_idx]} of recording"
    return "spread throughout recording"


def classify_gaps(summary: dict, gaps: list[dict], expected_duration: float | None = None) -> str:
    """Classify a stream's gap summary into a human-readable verdict, optionally
    comparing its time span against an expected value (e.g. the max duration across
    streams in the file, to flag streams that cut off early). Duration is used rather
    than raw sample count so streams with different sampling rates can be compared directly."""
    if summary["n_samples"] < 2:
        return "NO DATA"
    if expected_duration is not None and expected_duration > 0 and summary["duration"] < MIN_DURATION_FRACTION * expected_duration:
        return "CORRUPT / TOO SHORT"
    # Sustained rate mismatch with no discrete gaps: every interval is a bit off nominal,
    # rather than a few isolated dropouts -- treat separately from the "ok" case since it
    # usually indicates a real clock deviation rather than dropped samples.
    if summary["n_gaps"] == 0:
        if summary["nominal_srate"] > 0:
            rate_ratio = summary["effective_srate"] / summary["nominal_srate"]
            if abs(1 - rate_ratio) > RATE_MISMATCH_TOLERANCE:
                return "SUSTAINED RATE MISMATCH (no discrete gaps, likely real clock deviation)"
        if summary["jitter_pct"] > JITTER_THRESHOLD_PCT:
            return f"HIGH JITTER (std={summary['jitter_pct']:.1f}% of nominal period, no discrete gaps)"
        return "ok"
    missing_fraction = summary["total_missing"] / summary["n_samples"]
    if missing_fraction < NEGLIGIBLE_MISSING_FRACTION:
        return "ok (negligible gaps)"
    location = describe_gap_distribution(gaps, summary["duration"])
    if summary["n_gaps"] <= SALVAGEABLE_MAX_GAPS and missing_fraction < SALVAGEABLE_MISSING_FRACTION:
        return f"salvageable (few concentrated gaps, {location})"
    return f"SEVERE (many/large gaps, {location})"


def validate_stream(stream: XDFStream, expected_duration: float | None = None) -> dict:
    """Run the sample-timing checks on a stream and return its result row."""
    if not stream.is_regular:
        duration = stream.duration
        if expected_duration is not None and expected_duration > 0 and duration < MIN_DURATION_FRACTION * expected_duration:
            verdict = "CORRUPT / TOO SHORT"
        else:
            verdict = "irregular stream (skipped gap check)"
        return {
            "name": stream.name,
            "type": stream.type,
            "hostname": stream.hostname,
            "n_samples": stream.n_samples,
            "duration": duration,
            "effective_srate": 0.0,
            "nominal_srate": 0.0,
            "n_gaps": 0,
            "total_missing": 0,
            "jitter_std": 0.0,
            "jitter_pct": 0.0,
            "gaps": [],
            "verdict": verdict,
        }

    summary = summarize_gaps(stream.time_stamps, stream.nominal_srate)
    gaps = locate_gaps(stream.time_stamps, stream.nominal_srate)
    return {
        "name": stream.name,
        "type": stream.type,
        "hostname": stream.hostname,
        "verdict": classify_gaps(summary, gaps, expected_duration=expected_duration),
        "gaps": gaps,
        **summary,
    }


def validate_streams(path: str, streams: list[XDFStream]) -> dict:
    """Run sample-timing validation checks across a file's (already selected) streams.
    Returns a result dict describing each stream's validation outcome and an overall pass/fail status."""
    result = {
        "file": path,
        "error": None,
        "streams": [],
        "passed": False,
    }
    if not streams:
        result["error"] = "No matching streams found in file"
        return result

    expected_duration = max((stream.duration for stream in streams), default=0.0)
    result["streams"] = [validate_stream(stream, expected_duration) for stream in streams]
    result["passed"] = not any(row["verdict"].startswith(FAIL_PREFIXES) for row in result["streams"])
    return result


def validate_xdf_file(
        xdf_file: str,
        stream_type: str | None = None,
        exclude_name_substring: str | None = None,
    ) -> dict:
    """Load and validate a single XDF file. Returns a result dict describing each stream's
    validation outcome and an overall pass/fail status."""
    from xdf_tools.file import \
        XDFFile  # imported here to avoid a circular import

    try:
        xdf = XDFFile(xdf_file, synchronize_clocks=True, verbose=False)
    except Exception as exc:
        return {
            "file": xdf_file,
            "error": f"Failed to load XDF file: {exc}",
            "streams": [],
            "passed": False,
        }
    return xdf.validate(stream_type=stream_type, exclude_name_substring=exclude_name_substring)


def format_report(results: list[dict]) -> str:
    """Render validation results for one or more files as a human-readable text report."""
    header = (f"{'stream':<24} {'type':<10} {'hostname':<16} {'n_samples':>10} {'dur(s)':>8} "
              f"{'eff_Hz':>8} {'n_gaps':>7} {'missing':>8} {'jitter%':>8}  verdict")
    rule = "-" * len(header)

    lines = []
    n_passed = 0
    for result in results:
        lines.append(f"\n=== {result['file']} ===")
        if result["error"]:
            lines.append(f"  FAILED: {result['error']}")
            continue

        lines.append(header)
        lines.append(rule)
        for row in result["streams"]:
            lines.append(
                f"{row['name']:<24.24} {row['type']:<10.10} {row['hostname']:<16.16} "
                f"{row['n_samples']:>10} {row['duration']:>8.1f} {row['effective_srate']:>8.1f} "
                f"{row['n_gaps']:>7} {row['total_missing']:>8} {row['jitter_pct']:>8.2f}  {row['verdict']}"
            )
        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            n_passed += 1
        lines.append(f"  --> {status}")

    lines.append(f"\n{n_passed}/{len(results)} file(s) passed validation")
    return "\n".join(lines)
