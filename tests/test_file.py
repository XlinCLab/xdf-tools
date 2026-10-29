import pytest

from xdf_tools.file import XDFFile
from xdf_tools.stream import XDFStream


class SubStream(XDFStream):
    pass


def test_loads_all_streams_with_synchronized_clocks_by_default(recording):
    file = XDFFile(recording)
    assert file.clocks_synced
    assert len(file.streams) == 6
    assert all(stream._clocks_synced for stream in file.streams)
    assert repr(file) == f"XDFFile(path={recording!r}, n_streams=6)"


def test_unsynchronized_load_is_tracked(recording):
    file = XDFFile(recording, synchronize_clocks=False)
    assert not file.clocks_synced
    assert not any(stream._clocks_synced for stream in file.streams)


def test_synchronization_changes_timestamps(recording):
    synced = XDFFile(recording).stream_by_name("EEG-Device")
    raw = XDFFile(recording, synchronize_clocks=False).stream_by_name("EEG-Device")
    assert abs(synced.start_time - raw.start_time) > 1000


def test_streams_by_name_returns_all_matches(recording):
    matches = XDFFile(recording).streams_by_name("pupil_capture")
    assert [stream.hostname for stream in matches] == ["host-b", "host-c"]


def test_stream_by_name_returns_first_match(recording):
    assert XDFFile(recording).stream_by_name("pupil_capture").hostname == "host-b"


def test_stream_by_name_missing_raises(recording):
    with pytest.raises(ValueError, match="No <nope> stream found"):
        XDFFile(recording).stream_by_name("nope")


def test_stream_by_name_as_subclass(recording):
    stream = XDFFile(recording).stream_by_name("audio", stream_class=SubStream)
    assert type(stream) is SubStream
    assert stream.name == "audio"


def test_streams_by_type_is_case_insensitive(recording):
    assert [s.name for s in XDFFile(recording).streams_by_type("eeg")] == ["EEG-Device", "EEG-Device Impedance"]


def test_streams_by_type_all_types_by_default(recording):
    assert len(XDFFile(recording).streams_by_type()) == 6


def test_streams_by_type_excludes_name_substring(recording):
    streams = XDFFile(recording).streams_by_type("EEG", exclude_name_substring="impedance")
    assert [s.name for s in streams] == ["EEG-Device"]
    assert len(XDFFile(recording).streams_by_type(exclude_name_substring="Impedance")) == 5


def test_stream_by_type(recording):
    file = XDFFile(recording)
    assert file.stream_by_type("EEG").name == "EEG-Device"
    assert file.stream_by_type("EEG", stream_class=SubStream).name == "EEG-Device"
    with pytest.raises(ValueError, match="No <video> stream found"):
        file.stream_by_type("video")


def test_validate_all_streams(recording):
    result = XDFFile(recording).validate(exclude_name_substring="Impedance")
    assert result["error"] is None
    assert [row["name"] for row in result["streams"]] == ["audio", "pupil_capture", "pupil_capture", "EEG-Device", "Markers"]
    assert result["passed"]


def test_validate_no_matching_streams(recording):
    result = XDFFile(recording).validate(stream_type="video")
    assert result["error"] == "No matching streams found in file"
    assert not result["passed"]
