"""Fixtures generating small synthetic XDF recordings that mimic a multi-device LSL session:
streams recorded on different hosts, each with its own raw local clock and a linear clock drift
relative to the shared clock."""
from dataclasses import dataclass

import numpy as np
import pytest

from tests.xdf_writer import SyntheticStream, write_xdf

SYNCED_START = 1000.0  # shared-clock time at which every stream starts recording
DURATION = 20.0  # seconds


@dataclass
class ClockModel:
    """synced = raw + intercept + slope * raw, with the raw (local) clock of a stream starting at `raw_start`."""
    raw_start: float
    slope: float

    @property
    def intercept(self) -> float:
        return SYNCED_START - self.raw_start - self.slope * self.raw_start


AUDIO_CLOCK = ClockModel(raw_start=SYNCED_START, slope=3e-6)
EYETRACKER_CLOCK = ClockModel(raw_start=50_000.0, slope=-5e-6)
EEG_CLOCK = ClockModel(raw_start=7_000.0, slope=8e-6)
OTHER_EYETRACKER_CLOCK = ClockModel(raw_start=90_000.0, slope=2e-6)


def make_clock_measurements(model: ClockModel, rng, spacing: float = 1.0, noise: float = 2e-6):
    """Clock-offset measurements (as LSL's periodic time_correction pings would give), with some noise."""
    clock_times = model.raw_start - 1.0 + np.arange(0.0, DURATION + 2.0, spacing)
    clock_values = model.intercept + model.slope * clock_times + rng.normal(0, noise, len(clock_times))
    return clock_times, clock_values


def make_regular_stream(name, stream_type, hostname, srate, model, rng, n_channels, channel_format="float32",
                        labels=None, units=None, jitter=0.0, amplitude=1.0):
    n_samples = int(DURATION * srate)
    time_stamps = model.raw_start + np.arange(n_samples) / srate
    if jitter:
        time_stamps = time_stamps + rng.normal(0, jitter, n_samples)
    t = np.arange(n_samples) / srate
    data = np.stack([amplitude * np.sin(2 * np.pi * (3 + ch) * t) for ch in range(n_channels)], axis=1)
    if channel_format == "int16":
        data = np.round(data * 10000)
    clock_times, clock_values = make_clock_measurements(model, rng)
    return SyntheticStream(
        name=name,
        type=stream_type,
        hostname=hostname,
        nominal_srate=srate,
        channel_format=channel_format,
        time_stamps=time_stamps,
        data=data,
        labels=labels,
        units=units,
        clock_times=clock_times,
        clock_values=clock_values,
    )


def make_irregular_stream(name, stream_type, hostname, model, rng, n_samples=100, labels=None):
    time_stamps = np.sort(model.raw_start + rng.uniform(0.5, DURATION - 0.5, n_samples))
    n_channels = len(labels) if labels else 1
    data = rng.uniform(0, 1, (n_samples, n_channels))
    clock_times, clock_values = make_clock_measurements(model, rng)
    return SyntheticStream(
        name=name,
        type=stream_type,
        hostname=hostname,
        nominal_srate=0.0,
        channel_format="float32",
        time_stamps=time_stamps,
        data=data,
        labels=labels,
        clock_times=clock_times,
        clock_values=clock_values,
    )


@pytest.fixture(scope="session")
def recording(tmp_path_factory):
    """A synthetic recording; returns its path.
    Streams:
    audio (2 ch int16, 2 kHz, host-a)
    pupil_capture (irregular, host-b and host-c)
    EEG (4 ch, 250 Hz with timestamp jitter, host-b)
    EEG Impedance (irregular, host-b), Markers (string, host-a)
    """
    rng = np.random.default_rng(0)
    markers = SyntheticStream(
        name="Markers",
        type="Markers",
        hostname="host-a",
        nominal_srate=0.0,
        channel_format="string",
        time_stamps=AUDIO_CLOCK.raw_start + np.array([1.0, 5.0, 12.5]),
        data=[["start"], ["stimulus"], ["end"]],
        clock_times=make_clock_measurements(AUDIO_CLOCK, rng)[0],
        clock_values=make_clock_measurements(AUDIO_CLOCK, rng)[1],
    )
    streams = [
        make_regular_stream(
            name="audio",
            stream_type="Audio",
            hostname="host-a",
            srate=2000.0,
            model=AUDIO_CLOCK,
            rng=rng,
            n_channels=2,
            channel_format="int16",
        ),
        make_irregular_stream(
            name="pupil_capture",
            stream_type="Gaze",
            hostname="host-b",
            model=EYETRACKER_CLOCK,
            rng=rng,
            labels=["confidence", "norm_pos_x", "norm_pos_y"],
        ),
        make_irregular_stream(
            name="pupil_capture",
            stream_type="Gaze",
            hostname="host-c",
            model=OTHER_EYETRACKER_CLOCK,
            rng=rng,
            labels=["confidence", "norm_pos_x", "norm_pos_y"],
        ),
        make_regular_stream(
            name="EEG-Device",
            stream_type="EEG",
            hostname="host-b",
            srate=250.0,
            model=EEG_CLOCK,
            rng=rng,
            n_channels=4,
            labels=["Fz", "Cz", "Pz", "Oz"],
            units=["microvolts"] * 4,
            jitter=2e-4,
            amplitude=50.0,
        ),
        make_irregular_stream(
            name="EEG-Device Impedance",
            stream_type="EEG",
            hostname="host-b",
            model=EEG_CLOCK,
            rng=rng,
            n_samples=10,
            labels=["Fz", "Cz", "Pz", "Oz"][:1],
        ),
        markers,
    ]
    path = tmp_path_factory.mktemp("recordings") / "synthetic.xdf"
    write_xdf(path, streams)
    return str(path)
