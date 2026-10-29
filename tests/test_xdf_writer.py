"""Sanity check of the synthetic recording itself, so that failures in other tests can be told apart from fixture problems."""
import numpy as np
import pyxdf


def test_recording_loads_with_pyxdf(recording):
    streams, _ = pyxdf.load_xdf(recording, synchronize_clocks=False, dejitter_timestamps=False)
    assert [s["info"]["name"][0] for s in streams] == [
        "audio",
        "pupil_capture",
        "pupil_capture",
        "EEG-Device",
        "EEG-Device Impedance",
        "Markers",
    ]
    audio = streams[0]
    assert audio["time_series"].shape == (40000, 2)
    assert audio["time_series"].dtype == np.int16
    assert len(audio["clock_times"]) > 10
    assert streams[-1]["time_series"] == [["start"], ["stimulus"], ["end"]]
