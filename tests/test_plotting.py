import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from xdf_tools.file import XDFFile
from xdf_tools.plotting import (UNIT_ABBREVIATIONS, _format_unit,
                                build_stream_plot, plot_xdf_file)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_format_unit():
    assert _format_unit("") == ""
    assert _format_unit("Microvolts") == UNIT_ABBREVIATIONS["microvolts"]
    assert _format_unit("furlongs") == "furlongs"


def test_one_panel_per_channel_of_regular_streams_and_per_irregular_stream(qapp, recording):
    xdf = XDFFile(recording)
    streams = [xdf.stream_by_name("EEG-Device"), xdf.stream_by_name("pupil_capture"), xdf.stream_by_name("audio")]
    widget = build_stream_plot(streams)
    assert len(widget.ci.items) == 4 + 1 + 2


def test_channel_selection(qapp, recording):
    xdf = XDFFile(recording)
    widget = build_stream_plot([xdf.stream_by_name("EEG-Device")], channel_selections=[[0, 2]])
    assert len(widget.ci.items) == 2


def test_plot_xdf_file_filters(qapp, recording):
    widget = plot_xdf_file(recording, stream_type="EEG", exclude_name_substring="Impedance")
    assert len(widget.ci.items) == 4
    widget = plot_xdf_file(recording, stream_names=["audio"])
    assert len(widget.ci.items) == 2


def test_no_streams_raises(qapp):
    with pytest.raises(ValueError, match="No streams to plot"):
        build_stream_plot([])
