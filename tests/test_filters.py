from xdf_tools.file import XDFFile
from xdf_tools.filters import (exclude_streams_by_name_substring,
                               filter_streams_by_hostname,
                               filter_streams_by_type)


def test_filter_streams_by_hostname(recording):
    streams = XDFFile(recording).streams_by_name("pupil_capture")
    assert [s.hostname for s in filter_streams_by_hostname(streams, "host-c")] == ["host-c"]


def test_filter_streams_by_hostname_is_case_insensitive_and_accepts_several_hosts(recording):
    streams = XDFFile(recording).streams
    assert len(filter_streams_by_hostname(streams, "HOST-A")) == 2
    assert {s.hostname for s in filter_streams_by_hostname(streams, ["Host-B", "host-c"])} == {"host-b", "host-c"}


def test_filter_streams_by_hostname_no_match(recording):
    assert filter_streams_by_hostname(XDFFile(recording).streams, "other") == []
    assert filter_streams_by_hostname(XDFFile(recording).streams, []) == []


def test_filter_streams_by_type(recording):
    streams = XDFFile(recording).streams
    assert len(filter_streams_by_type(streams, "gaze")) == 2
    assert filter_streams_by_type(streams, None) == streams


def test_exclude_streams_by_name_substring(recording):
    streams = XDFFile(recording).streams
    assert len(exclude_streams_by_name_substring(streams, "IMPEDANCE")) == 5
    assert exclude_streams_by_name_substring(streams, None) == streams
