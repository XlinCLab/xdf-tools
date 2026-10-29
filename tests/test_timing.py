import numpy as np
import pytest

from tests.conftest import AUDIO_CLOCK, make_regular_stream
from tests.xdf_writer import SyntheticStream, write_xdf
from xdf_tools.file import XDFFile
from xdf_tools.timing import (classify_gaps, describe_gap_distribution,
                              format_report, locate_gaps, summarize_gaps,
                              validate_xdf_file)

SRATE = 100.0


def regular_timestamps(n=1000, srate=SRATE, drop=(), scale=1.0):
    """Timestamps of n regularly spaced samples (period scaled by `scale`), minus the dropped sample indices."""
    time_stamps = np.arange(n) / srate * scale
    return np.delete(time_stamps, list(drop))


def verdict(time_stamps, srate=SRATE, expected_duration=None):
    summary = summarize_gaps(time_stamps, srate)
    return classify_gaps(summary, locate_gaps(time_stamps, srate), expected_duration)


def test_summarize_gaps_without_gaps():
    summary = summarize_gaps(regular_timestamps(), SRATE)
    assert summary["n_samples"] == 1000
    assert summary["n_gaps"] == 0
    assert summary["total_missing"] == 0
    assert summary["effective_srate"] == pytest.approx(SRATE)
    assert summary["jitter_pct"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_gaps_counts_dropped_samples():
    summary = summarize_gaps(regular_timestamps(drop=[100, 101, 102, 500]), SRATE)
    assert summary["n_gaps"] == 2
    assert summary["total_missing"] == 4
    assert summary["max_gap_missing"] == 3


def test_summarize_gaps_irregular_or_short_streams():
    assert summarize_gaps(np.array([1.0, 2.0, 5.0]), 0.0)["n_gaps"] == 0
    summary = summarize_gaps(np.array([1.0]), SRATE)
    assert summary["n_samples"] == 1
    assert summary["duration"] == 0.0


def test_locate_gaps_is_relative_to_first_timestamp():
    time_stamps = regular_timestamps(drop=[200, 201]) + 500.0
    gaps = locate_gaps(time_stamps, SRATE)
    assert len(gaps) == 1
    assert gaps[0]["start"] == pytest.approx(1.99)
    assert gaps[0]["end"] == pytest.approx(2.02)
    assert gaps[0]["n_missing"] == 2
    assert locate_gaps(np.array([1.0, 2.0]), 0.0) == []


def test_describe_gap_distribution():
    assert describe_gap_distribution([], 10.0) == "n/a"
    concentrated = [{"start": 0.5, "end": 0.6, "n_missing": 5}, {"start": 1.0, "end": 1.1, "n_missing": 5}]
    assert describe_gap_distribution(concentrated, 10.0) == "concentrated in first third of recording"
    spread = [{"start": 1.0, "end": 1.1, "n_missing": 5}, {"start": 5.0, "end": 5.1, "n_missing": 5},
              {"start": 9.0, "end": 9.1, "n_missing": 5}]
    assert describe_gap_distribution(spread, 10.0) == "spread throughout recording"


def test_classify_ok():
    assert verdict(regular_timestamps()) == "ok"


def test_classify_negligible_gaps():
    assert verdict(regular_timestamps(n=2000, drop=[1000])) == "ok (negligible gaps)"


def test_classify_salvageable():
    assert verdict(regular_timestamps(n=1000, drop=range(100, 140))).startswith("salvageable")


def test_classify_severe():
    drops = [i for i in range(0, 900, 100) for i in (i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6)]
    assert verdict(regular_timestamps(n=1000, drop=drops)).startswith("SEVERE")


def test_classify_sustained_rate_mismatch():
    assert verdict(regular_timestamps(scale=1.05)).startswith("SUSTAINED RATE MISMATCH")


def test_classify_high_jitter():
    rng = np.random.default_rng(0)
    time_stamps = np.cumsum(rng.uniform(0.004, 0.0159, 2000))  # mean period ~ nominal but irregular
    time_stamps = np.cumsum(np.full(2000, 0.01) + rng.uniform(-0.0045, 0.0045, 2000))
    assert verdict(time_stamps).startswith("HIGH JITTER")


def test_classify_no_data_and_too_short():
    assert verdict(np.array([1.0])) == "NO DATA"
    assert verdict(regular_timestamps(n=100), expected_duration=100.0) == "CORRUPT / TOO SHORT"
    assert verdict(regular_timestamps(n=100), expected_duration=1.5) == "ok"


def test_synthetic_recording_streams_validate_ok(recording):
    xdf = XDFFile(recording)
    for stream in xdf.streams_by_type("EEG", exclude_name_substring="Impedance") + [xdf.stream_by_name("audio")]:
        assert stream.classify_gaps() == "ok", stream
    assert xdf.stream_by_name("audio").summarize_gaps()["n_gaps"] == 0
    assert xdf.stream_by_name("audio").locate_gaps() == []
    assert xdf.stream_by_name("audio").gap_distribution() == "n/a"


@pytest.fixture
def faulty_recording(tmp_path):
    """EEG-like stream with a dropped block of samples and a second stream that stops early.
    NB: pyxdf's default timestamp dejittering only breaks segments at gaps longer than max(1 s, 500 samples),
    and smooths over shorter dropouts, so the gap must be longer than that to be visible."""
    rng = np.random.default_rng(5)
    gappy = make_regular_stream("gappy", "EEG", "host-a", 250.0, AUDIO_CLOCK, rng, 1)
    keep = np.ones(len(gappy.time_stamps), dtype=bool)
    keep[2000:2600] = False
    gappy.time_stamps, gappy.data = gappy.time_stamps[keep], gappy.data[keep]
    short = make_regular_stream("short", "EEG", "host-a", 250.0, AUDIO_CLOCK, rng, 1)
    short.time_stamps, short.data = short.time_stamps[:500], short.data[:500]
    healthy = make_regular_stream("healthy", "EEG", "host-a", 250.0, AUDIO_CLOCK, rng, 1)
    path = tmp_path / "faulty.xdf"
    write_xdf(path, [gappy, short, healthy])
    return str(path)


def test_validate_flags_gappy_and_short_streams(faulty_recording):
    result = XDFFile(faulty_recording).validate()
    rows = {row["name"]: row for row in result["streams"]}
    assert rows["healthy"]["verdict"] == "ok"
    assert rows["gappy"]["verdict"].startswith("salvageable")
    assert rows["gappy"]["n_gaps"] == 1 and rows["gappy"]["total_missing"] == 600
    assert rows["gappy"]["gaps"][0]["n_missing"] == 600
    assert rows["short"]["verdict"] == "CORRUPT / TOO SHORT"
    assert not result["passed"]


def test_validate_xdf_file(recording, faulty_recording):
    assert validate_xdf_file(recording)["passed"]
    assert not validate_xdf_file(faulty_recording)["passed"]
    assert validate_xdf_file(faulty_recording, stream_type="EEG", exclude_name_substring="short")["passed"]
    result = validate_xdf_file(recording, stream_type="EEG", exclude_name_substring="Impedance")
    assert [row["name"] for row in result["streams"]] == ["EEG-Device"]


def test_validate_xdf_file_unreadable_file(tmp_path):
    result = validate_xdf_file(str(tmp_path / "missing.xdf"))
    assert not result["passed"]
    assert result["error"].startswith("Failed to load XDF file")


def test_validate_irregular_stream_too_short_vs_skipped(tmp_path):
    long_stream = make_regular_stream("long", "EEG", "host-a", 100.0, AUDIO_CLOCK, np.random.default_rng(0), 1)
    events = SyntheticStream(
        name="events", type="Markers", hostname="host-a", nominal_srate=0.0, channel_format="float32",
        time_stamps=AUDIO_CLOCK.raw_start + np.array([1.0, 2.0]), data=np.zeros((2, 1)),
        clock_times=long_stream.clock_times, clock_values=long_stream.clock_values,
    )
    path = tmp_path / "events.xdf"
    write_xdf(path, [long_stream, events])
    rows = {row["name"]: row for row in XDFFile(str(path)).validate()["streams"]}
    assert rows["events"]["verdict"] == "CORRUPT / TOO SHORT"
    rows = {row["name"]: row for row in XDFFile(str(path)).validate(stream_type="Markers")["streams"]}
    assert rows["events"]["verdict"] == "irregular stream (skipped gap check)"


def test_format_report(faulty_recording, recording):
    report = format_report([XDFFile(faulty_recording).validate(), XDFFile(recording).validate()])
    assert f"=== {faulty_recording} ===" in report
    assert "CORRUPT / TOO SHORT" in report
    assert "--> FAIL" in report and "--> PASS" in report
    assert report.endswith("1/2 file(s) passed validation")
    error_report = format_report([{"file": "x.xdf", "error": "boom", "streams": [], "passed": False}])
    assert "FAILED: boom" in error_report
