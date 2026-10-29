import numpy as np
import pytest

from tests.conftest import AUDIO_CLOCK, DURATION, SYNCED_START
from xdf_tools.file import XDFFile
from xdf_tools.stream import XDFStream


def make_stream_dict(time_series, time_stamps=None, labels=None, srate="100", fmt="float32", **info):
    desc = []
    if labels is not None:
        desc = [{"channels": [{"channel": [{"label": [label]} for label in labels]}]}]
    return {
        "info": {
            "name": ["s"],
            "type": ["T"],
            "hostname": ["h"],
            "nominal_srate": [srate],
            "channel_format": [fmt],
            "desc": desc,
            **info
        },
        "time_series": time_series,
        "time_stamps": np.array([] if time_stamps is None else time_stamps),
    }


def test_metadata(recording):
    audio = XDFFile(recording).stream_by_name("audio")
    assert (audio.name, audio.type, audio.hostname) == ("audio", "Audio", "host-a")
    assert audio.nominal_srate == 2000.0
    assert audio.is_regular
    assert audio.channel_format == "int16"
    assert audio.n_channels == 2
    assert audio.n_samples == int(DURATION * 2000)
    assert repr(audio) == "XDFStream(name='audio', type='Audio', hostname='host-a')"


def test_irregular_stream(recording):
    gaze = XDFFile(recording).stream_by_name("pupil_capture")
    assert not gaze.is_regular
    assert gaze.channel_labels == ["confidence", "norm_pos_x", "norm_pos_y"]
    assert gaze.channel_units == ["", "", ""]
    assert gaze.channel_field("label") == gaze.channel_labels


def test_channel_labels_and_units(recording):
    eeg = XDFFile(recording).stream_by_name("EEG-Device")
    assert eeg.channel_labels == ["Fz", "Cz", "Pz", "Oz"]
    assert eeg.channel_units == ["microvolts"] * 4


def test_channel_labels_fall_back_when_undeclared(recording):
    audio = XDFFile(recording).stream_by_name("audio")
    assert audio.channel_labels == ["Ch 0", "Ch 1"]
    assert audio.channel_units == ["", ""]


def test_timing_properties_synced(recording):
    audio = XDFFile(recording).stream_by_name("audio")
    assert audio.start_time == pytest.approx(SYNCED_START, abs=1e-3)
    assert audio.end_time - audio.start_time == pytest.approx(DURATION, abs=0.01)
    assert audio.duration == audio.end_time - audio.start_time
    assert audio.relative_times[0] == 0.0
    assert audio.relative_times[-1] == pytest.approx(audio.duration)


def test_time_series_is_samples_by_channels(recording):
    file = XDFFile(recording)
    assert file.stream_by_name("audio").time_series.shape == (40000, 2)
    assert file.stream_by_name("EEG-Device").time_series.shape == (5000, 4)


def test_time_series_transposed_input_is_corrected_using_channel_labels():
    stream = XDFStream(make_stream_dict(np.zeros((3, 50)), labels=["a", "b", "c"]))
    assert stream.time_series.shape == (50, 3)
    assert stream.n_channels == 3


def test_time_series_1d_input_becomes_single_channel():
    stream = XDFStream(make_stream_dict(np.arange(10.0)))
    assert stream.time_series.shape == (10, 1)


def test_empty_stream_start_time_raises():
    stream = XDFStream(make_stream_dict(np.empty((0, 1))))
    assert stream.n_samples == 0
    assert stream.duration == 0.0
    with pytest.raises(ValueError, match="no samples"):
        stream.start_time
    with pytest.raises(ValueError, match="no samples"):
        stream.end_time
    with pytest.raises(ValueError, match="no samples"):
        stream.index_at(np.array([0.0]))


def test_footer_info(recording):
    stream = XDFFile(recording, synchronize_clocks=False).stream_by_name("pupil_capture")
    assert stream.raw_start_time == stream.start_time
    assert stream.raw_end_time == stream.end_time


def test_footer_info_missing_raises():
    stream = XDFStream(make_stream_dict(np.zeros((2, 1)), time_stamps=[0.0, 1.0]))
    with pytest.raises(ValueError, match="first_timestamp"):
        stream.raw_start_time


def test_index_at_regular_stream_interpolates_and_extrapolates_at_nominal_rate():
    stream = XDFStream(make_stream_dict(np.zeros((5, 1)), time_stamps=[10.0, 10.01, 10.02, 10.03, 10.04], srate="100"))
    np.testing.assert_allclose(stream.index_at(np.array([10.0, 10.025, 10.04])), [0.0, 2.5, 4.0])
    np.testing.assert_allclose(stream.index_at(np.array([9.98, 10.06])), [-2.0, 6.0])


def test_index_at_irregular_stream_clamps():
    stream = XDFStream(make_stream_dict(np.zeros((3, 1)), time_stamps=[1.0, 2.0, 4.0], srate="0"))
    np.testing.assert_allclose(stream.index_at(np.array([0.0, 3.0, 9.0])), [0.0, 1.5, 2.0])


def test_from_stream_preserves_clock_state_and_subclass(recording):
    unsynced = XDFFile(recording, synchronize_clocks=False).stream_by_name("audio")

    class XdfSubclass(XDFStream):
        pass

    converted = XdfSubclass.from_stream(unsynced)
    assert type(converted) is XdfSubclass
    assert converted._stream is unsynced._stream
    assert not converted._clocks_synced
    assert repr(converted).startswith("XdfSubclass(")


def test_string_stream(recording):
    markers = XDFFile(recording).stream_by_name("Markers")
    assert not markers.is_regular
    assert markers.time_series.tolist() == [["start"], ["stimulus"], ["end"]]
    assert markers.channel_format == "string"


def test_stream_synced_start_matches_the_simulated_synchronization(recording):
    """Every stream starts at the same shared-clock time by construction of the fixture, despite their different raw clocks."""
    file = XDFFile(recording)
    raw = XDFFile(recording, synchronize_clocks=False)
    for stream in file.streams:
        if stream.name in ("audio", "EEG-Device"):
            assert stream.start_time == pytest.approx(SYNCED_START, abs=1e-3)
    assert raw.stream_by_name("EEG-Device").start_time == pytest.approx(7000.0, abs=1e-2)
    assert AUDIO_CLOCK.raw_start == SYNCED_START
