import numpy as np
import pytest

from tests.conftest import (AUDIO_CLOCK, DURATION, EEG_CLOCK, EYETRACKER_CLOCK,
                            SYNCED_START, make_clock_measurements,
                            make_regular_stream)
from tests.xdf_writer import write_xdf
from xdf_tools.clock import (ClockResetError, apply_clock_drift_correction,
                             convert_relative_time_to_synced, fit_linear_drift,
                             summarize_clock_offsets)
from xdf_tools.file import XDFFile

CLOCKS = {
    "audio": AUDIO_CLOCK,
    "EEG-Device": EEG_CLOCK,
}


@pytest.mark.parametrize("raw_start", [1.6e9, 1.5e5])
@pytest.mark.parametrize("intercept", [-55.09, 0.0, 1.67e9])
@pytest.mark.parametrize("slope", [3.3e-8, -2.1e-8, 5.5e-6])
@pytest.mark.parametrize("relative_time", [0.0, 1.0, 600.5])
def test_convert_relative_time_to_synced_matches_apply_clock_drift_correction(raw_start, intercept, slope, relative_time):
    """For a stream with an exactly regular raw clock, converting a relative time anchored on the synced start
    agrees with applying the drift correction to the corresponding raw time."""
    synced_start = apply_clock_drift_correction(raw_start, intercept, slope)
    expected = apply_clock_drift_correction(raw_start + relative_time, intercept, slope)
    np.testing.assert_allclose(convert_relative_time_to_synced(relative_time, synced_start, slope), expected, rtol=0, atol=1e-5)


def test_convert_relative_time_to_synced_at_zero_returns_synced_start_time_exactly():
    assert convert_relative_time_to_synced(0.0, 1669975520.1383622, 3.3e-8) == 1669975520.1383622


def test_fit_linear_drift_recovers_known_correction():
    rng = np.random.default_rng(1)
    clock_times, clock_values = make_clock_measurements(EYETRACKER_CLOCK, rng, noise=1e-6)
    intercept, slope = fit_linear_drift(clock_times, clock_values)
    assert slope == pytest.approx(EYETRACKER_CLOCK.slope, abs=1e-7)
    assert intercept + slope * clock_times[0] == pytest.approx(clock_values[0], abs=1e-4)


def test_fit_linear_drift_requires_two_measurements():
    with pytest.raises(ValueError, match="At least 2"):
        fit_linear_drift(np.array([1.0]), np.array([0.1]))


@pytest.mark.parametrize("stream_name", ["audio", "pupil_capture", "EEG-Device"])
def test_fit_drift_correction_reproduces_pyxdf_synchronization(recording, stream_name):
    """The fit must reproduce the correction pyxdf itself applied: the stream's raw (dejittered) timestamps,
    corrected with the fitted intercept and slope, must equal the timestamps pyxdf synchronized."""
    synced = XDFFile(recording, synchronize_clocks=True).stream_by_name(stream_name)
    raw = XDFFile(recording, synchronize_clocks=False).stream_by_name(stream_name)

    intercept, slope = synced.fit_drift_correction()
    predicted = apply_clock_drift_correction(raw.time_stamps, intercept, slope)
    assert np.max(np.abs(predicted - synced.time_stamps)) < 1e-5


@pytest.mark.parametrize("stream_name", ["audio", "EEG-Device"])
def test_convert_relative_time_to_synced_matches_pyxdf_for_regular_rate_streams(recording, stream_name):
    """Times relative to the stream's start, converted from the synced start time and drift slope alone,
    must land on pyxdf's own synchronized timestamps."""
    synced = XDFFile(recording, synchronize_clocks=True).stream_by_name(stream_name)
    _, slope = synced.fit_drift_correction()
    converted = convert_relative_time_to_synced(
        synced.relative_times / (1 + slope), synced.start_time, slope
    )
    assert np.max(np.abs(converted - synced.time_stamps)) < 1e-5


def test_fit_drift_correction_recovers_simulated_drift(recording):
    """The simulated clock drift of the fixture is recovered from the file (slope of the synced-vs-raw relation)."""
    for stream_name, model in (("audio", AUDIO_CLOCK), ("EEG-Device", EEG_CLOCK)):
        intercept, slope = XDFFile(recording).stream_by_name(stream_name).fit_drift_correction()
        assert slope == pytest.approx(model.slope, abs=5e-7)
        assert apply_clock_drift_correction(model.raw_start, intercept, slope) == pytest.approx(SYNCED_START, abs=1e-3)


def test_raw_start_time_is_footer_value_not_dejittered_start(recording):
    """For a regular-rate stream with jittery timestamps, the raw start time read from the footer is the true first
    raw timestamp, which differs from the dejittered raw start pyxdf's drift correction is fit against."""
    raw_dejittered = XDFFile(recording, synchronize_clocks=False).stream_by_name("EEG-Device")
    raw_footer_start = raw_dejittered.raw_start_time
    assert raw_footer_start != raw_dejittered.start_time
    assert abs(raw_footer_start - raw_dejittered.start_time) < 1e-2
    # The footer is unaffected by clock synchronization
    synced = XDFFile(recording, synchronize_clocks=True).stream_by_name("EEG-Device")
    assert synced.raw_start_time == raw_footer_start


def test_clock_segments_single_segment(recording):
    for stream in XDFFile(recording).streams:
        assert len(stream.clock_segments) == 1


def test_clock_segments_empty_without_synchronization(recording):
    assert XDFFile(recording, synchronize_clocks=False).stream_by_name("audio").clock_segments == []


def test_fit_drift_correction_requires_synchronized_load(recording):
    stream = XDFFile(recording, synchronize_clocks=False).stream_by_name("audio")
    with pytest.raises(ValueError, match="synchronize_clocks=False"):
        stream.fit_drift_correction()


def test_fit_drift_correction_raises_on_clock_reset(tmp_path):
    """A mid-recording clock reset (a gap in the clock measurements plus a jump in the offset)
    splits the clock data into several segments, for which no single linear drift can be fit."""
    rng = np.random.default_rng(2)
    stream = make_regular_stream("audio", "Audio", "host-a", 100.0, AUDIO_CLOCK, rng, 1)
    stream.clock_times = AUDIO_CLOCK.raw_start + np.concatenate([np.arange(0, 7, 1.0), np.arange(13, 20, 1.0)])
    stream.clock_values = np.concatenate([np.full(7, 0.01), np.full(7, 30.0)]) + rng.normal(0, 1e-5, 14)
    path = tmp_path / "reset.xdf"
    write_xdf(path, [stream])

    loaded = XDFFile(str(path)).stream_by_name("audio")
    assert len(loaded.clock_segments) > 1
    with pytest.raises(ClockResetError, match="clock reset"):
        loaded.fit_drift_correction()
    assert issubclass(ClockResetError, ValueError)


def test_fit_drift_correction_requires_two_clock_measurements(tmp_path):
    rng = np.random.default_rng(3)
    stream = make_regular_stream("audio", "Audio", "host-a", 100.0, AUDIO_CLOCK, rng, 1)
    stream.clock_times = stream.clock_times[:1]
    stream.clock_values = stream.clock_values[:1]
    path = tmp_path / "one_offset.xdf"
    write_xdf(path, [stream])
    with pytest.raises(ValueError, match="fewer than 2"):
        XDFFile(str(path)).stream_by_name("audio").fit_drift_correction()


def test_summarize_clock_offsets():
    assert summarize_clock_offsets(np.empty(0)) == {"n_measurements": 0, "mean_offset": 0.0, "drift": 0.0}
    assert summarize_clock_offsets(np.array([0.5])) == {"n_measurements": 1, "mean_offset": 0.5, "drift": 0.0}
    summary = summarize_clock_offsets(np.array([1.0, 2.0, 4.0]))
    assert summary["n_measurements"] == 3
    assert summary["mean_offset"] == pytest.approx(7 / 3)
    assert summary["drift"] == 3.0


def test_stream_summarize_clock_offsets(recording):
    stream = XDFFile(recording).stream_by_name("audio")
    summary = stream.summarize_clock_offsets()
    assert summary["n_measurements"] == len(stream.clock_values)
    assert summary["drift"] == pytest.approx(AUDIO_CLOCK.slope * (len(stream.clock_values) - 1), abs=1e-4)
    assert DURATION > 0
