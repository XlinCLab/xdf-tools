"""XDFStream: a single stream loaded by pyxdf, with its metadata, timing, clock-synchronization data, and samples exposed as properties."""
from __future__ import annotations

import numpy as np

from xdf_tools import clock, timing
from xdf_tools.constants import (FOOTER_INFO_FIRST_TIMESTAMP,
                                 FOOTER_INFO_LAST_TIMESTAMP, INFO_CHANNEL,
                                 INFO_CHANNEL_FORMAT, INFO_CHANNELS,
                                 INFO_CLOCK_SEGMENTS, INFO_DESC, INFO_HOSTNAME,
                                 INFO_LABEL, INFO_NAME, INFO_NOMINAL_SRATE,
                                 INFO_TYPE, INFO_UNIT, STREAM_CLOCK_TIMES,
                                 STREAM_CLOCK_VALUES, STREAM_FOOTER,
                                 STREAM_INFO, STREAM_TIME_SERIES,
                                 STREAM_TIME_STAMPS)


class XDFStream:
    """Wraps a single stream dict as returned by `pyxdf.load_xdf()`.

    `clocks_synced` states whether the stream was loaded with `synchronize_clocks=True`,
    which determines the clock domain of `time_stamps` and whether `clock_segments` is populated.
    """

    def __init__(self, stream: dict, clocks_synced: bool = True):
        self._stream = stream
        self._clocks_synced = clocks_synced

    @classmethod
    def from_stream(cls, stream: XDFStream) -> XDFStream:
        """Return this same stream as an instance of a specified (sub)class."""
        return cls(stream._stream, clocks_synced=stream._clocks_synced)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self, field: str, result_type=None):
        """A field of the stream's `info` metadata (pyxdf wraps each value in a list)."""
        result = self._stream.get(STREAM_INFO, {}).get(field, [""])
        if isinstance(result, list):
            result = result[0] if result else ""
        if result_type is not None:
            return result_type(result)
        return result

    @property
    def name(self) -> str:
        return self.get_metadata(INFO_NAME, str)

    @property
    def type(self) -> str:
        return self.get_metadata(INFO_TYPE, str)

    @property
    def hostname(self) -> str:
        return self.get_metadata(INFO_HOSTNAME, str)

    @property
    def nominal_srate(self) -> float:
        return self.get_metadata(INFO_NOMINAL_SRATE, float)

    @property
    def channel_format(self) -> str:
        """Data type of the samples as declared by the stream, e.g. int16, float32, string."""
        return self.get_metadata(INFO_CHANNEL_FORMAT, str)

    @property
    def is_regular(self) -> bool:
        """Whether this stream has a fixed nominal sampling rate, as opposed to an
        irregular/event stream (e.g. markers) sampled asynchronously."""
        return self.nominal_srate > 0

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @property
    def time_stamps(self) -> np.ndarray:
        """This stream's per-sample timestamps, exactly as loaded (LSL-synchronized
        absolute seconds if the parent XDFFile was loaded with synchronize_clocks=True,
        else this stream's own raw local-clock seconds)."""
        return np.asarray(self._stream.get(STREAM_TIME_STAMPS, []), dtype=np.float64)

    @property
    def n_samples(self) -> int:
        return len(self.time_stamps)

    @property
    def duration(self) -> float:
        """Time span in seconds from the first to the last sample."""
        timestamps = self.time_stamps
        if len(timestamps) < 2:
            return 0.0
        return float(timestamps[-1] - timestamps[0])

    @property
    def start_time(self) -> float:
        """This stream's first sample's timestamp (see `time_stamps` for which clock
        domain this is in, depending on how the parent XDFFile was loaded)."""
        timestamps = self.time_stamps
        if len(timestamps) == 0:
            raise ValueError(f"Stream {self.name!r} has no samples")
        return float(timestamps[0])

    @property
    def end_time(self) -> float:
        """This stream's last sample's timestamp (see `time_stamps` for which clock
        domain this is in, depending on how the parent XDFFile was loaded)."""
        timestamps = self.time_stamps
        if len(timestamps) == 0:
            raise ValueError(f"Stream {self.name!r} has no samples")
        return float(timestamps[-1])

    @property
    def relative_times(self) -> np.ndarray:
        """This stream's timestamps relative to its start time (first timestamp)."""
        return self.time_stamps - self.start_time

    def index_at(self, times: np.ndarray) -> np.ndarray:
        """Fractional sample index at times (s), via the stream's own (dejittered) timestamps.

        Outside the recorded range this extrapolates with the nominal sampling rate of a regular stream;
        for an irregular stream the index is clamped to the first and last sample.
        """
        if self.n_samples == 0:
            raise ValueError(f"Stream {self.name!r} has no samples")
        times = np.asarray(times, dtype=np.float64)
        time_stamps = self.time_stamps
        indices = np.interp(times, time_stamps, np.arange(len(time_stamps), dtype=np.float64))
        if self.is_regular:
            before = times < time_stamps[0]
            after = times > time_stamps[-1]
            indices[before] = (times[before] - time_stamps[0]) * self.nominal_srate
            indices[after] = (len(time_stamps) - 1) + (times[after] - time_stamps[-1]) * self.nominal_srate
        return indices

    def get_footer_info(self, field: str) -> float:
        """Read field from the XDF footer chunk.
        The footer is parsed unconditionally and is never touched by clock synchronization,
        so timestamps listed here are always the stream's raw (un-synchronized) boundary timestamps,
        regardless of whether the parent XDFFile was loaded with synchronize_clocks=True."""
        footer_info = self._stream.get(STREAM_FOOTER, {}).get(STREAM_INFO, {})
        value = footer_info.get(field)
        if not value:
            raise ValueError(f"Stream {self.name!r} has no {field!r} in its XDF footer chunk")
        if isinstance(value, list):
            value = value[0]
        return float(value)

    @property
    def raw_start_time(self) -> float:
        """This stream's raw (un-synchronized, own local clock) first-sample timestamp, read from the XDF footer chunk.
        Due to timestamp dejittering for regular-rate streams, this is NOT safe to combine with apply_clock_drift_correction."""
        return self.get_footer_info(FOOTER_INFO_FIRST_TIMESTAMP)

    @property
    def raw_end_time(self) -> float:
        """This stream's raw (un-synchronized, own local clock) last-sample timestamp, read from the XDF footer chunk.
        Due to timestamp dejittering for regular-rate streams, this is NOT safe to combine with apply_clock_drift_correction."""
        return self.get_footer_info(FOOTER_INFO_LAST_TIMESTAMP)

    # ------------------------------------------------------------------
    # Clock synchronization
    # ------------------------------------------------------------------

    @property
    def clock_times(self) -> np.ndarray:
        """Local times (on this stream's own local clock) at which a clock-offset calibration
        measurement was taken during recording (periodic LSL clock-sync pings)."""
        return np.asarray(self._stream.get(STREAM_CLOCK_TIMES, []), dtype=np.float64)

    @property
    def clock_values(self) -> np.ndarray:
        """Measured clock offset at each of clock_times: the correction pyxdf applies
        (when loaded with synchronize_clocks=True) to map this stream's local clock onto
        the file's shared clock."""
        return np.asarray(self._stream.get(STREAM_CLOCK_VALUES, []), dtype=np.float64)

    @property
    def clock_segments(self) -> list[tuple[int, int]]:
        """Segment boundaries (start, end) index pairs, identified by pyxdf's clock-reset detection.
        A single segment means no clock discontinuity was detected during the recording.
        Only meaningful when the parent XDFFile was loaded with synchronize_clocks=True
        (empty otherwise, since reset detection only runs as part of that step)."""
        segments = self._stream.get(STREAM_INFO, {}).get(INFO_CLOCK_SEGMENTS, [])
        return [tuple(segment) for segment in segments]

    def summarize_clock_offsets(self) -> dict:
        """Summarize this stream's clock-offset calibration measurements."""
        return clock.summarize_clock_offsets(self.clock_values)

    def fit_drift_correction(self) -> tuple[float, float]:
        """Fit this stream's linear clock-offset correction (see `clock.fit_linear_drift`),
        making the drift correction intercept and slope available.

        Returns (intercept, slope), so that `synced = raw + intercept + slope * raw`.

        Raises `ClockResetError` if this stream's clock-offset data has more than one
        `clock_segments` entry (a detected mid-recording clock reset), since a single
        linear fit across a reset would be wrong. Also requires the `XDFFile` to have
        been loaded with `synchronize_clocks=True` (so that `clock_segments` is populated)."""
        if not self._clocks_synced:
            raise ValueError(
                f"Stream {self.name!r} was loaded with synchronize_clocks=False; "
                "clock_segments is only populated when synchronize_clocks=True."
            )
        segments = self.clock_segments
        if len(segments) != 1:
            raise clock.ClockResetError(
                f"Stream {self.name!r} has {len(segments)} clock segments (expected "
                "exactly 1), meaning that a mid-recording clock reset was detected, "
                "which drift correction does not currently handle."
            )
        if len(self.clock_times) < 2:
            raise ValueError(
                f"Stream {self.name!r} has fewer than 2 clock-offset measurements; "
                "cannot fit a drift correction."
            )
        return clock.fit_linear_drift(self.clock_times, self.clock_values)

    # ------------------------------------------------------------------
    # Samples and channels
    # ------------------------------------------------------------------

    def _raw_channel_field(self, field: str) -> list[str]:
        """
        Per-channel string field extraction from info.desc.channels.channel[]
        (e.g. "label" or "unit"), with no fallback -- used internally both
        to disambiguate time_series orientation and to look up channel labels/units.
        Returns [] if the metadata is absent or malformed (e.g. irregular
        streams typically don't declare per-channel metadata at all).
        """
        desc = self._stream.get(STREAM_INFO, {}).get(INFO_DESC, [])
        if isinstance(desc, list) and len(desc) > 0:
            desc = desc[0]
        channels = desc.get(INFO_CHANNELS, {}) if isinstance(desc, dict) else {}
        if isinstance(channels, list) and len(channels) > 0:
            channels = channels[0]
        channels = channels.get(INFO_CHANNEL, []) if isinstance(channels, dict) else []

        values = []
        for channel in channels:
            value = channel.get(field) if isinstance(channel, dict) else None
            if isinstance(value, list):
                value = value[0] if value else ""
            values.append(str(value) if value else "")
        return values

    @property
    def time_series(self) -> np.ndarray:
        """
        Sample data as a (n_samples, n_channels) array.
        pyxdf normally already returns this shape;
        the channel-label count (when available) is used to catch the rare
        case where it comes back transposed.
        """
        samples = np.asarray(self._stream.get(STREAM_TIME_SERIES, []))
        if samples.ndim == 1:
            samples = samples[:, None]
        labels = self._raw_channel_field(INFO_LABEL)
        if len(labels) and len(labels) == samples.shape[0] and len(labels) != samples.shape[1]:
            samples = samples.T
        return samples

    @property
    def n_channels(self) -> int:
        time_series = self.time_series
        return time_series.shape[1] if time_series.ndim == 2 else 1

    def channel_field(self, field: str) -> list[str]:
        """A per-channel metadata field (e.g. label, unit, coordinate_system), one string per channel
        (empty strings if the stream does not declare the field for every channel)."""
        values = self._raw_channel_field(field)
        n_channels = self.n_channels
        return values if len(values) == n_channels else [""] * n_channels

    @property
    def channel_labels(self) -> list[str]:
        """Per-channel labels, falling back to "Ch 0", "Ch 1", ... if the stream's
        metadata doesn't declare a usable label for every channel."""
        labels = self.channel_field(INFO_LABEL)
        if all(labels):
            return labels
        return [f"Ch {i}" for i in range(self.n_channels)]

    @property
    def channel_units(self) -> list[str]:
        """Per-channel unit strings (e.g. "microvolts", "kohms").
        Returns empty strings for channels without a declared unit."""
        return self.channel_field(INFO_UNIT)

    # ------------------------------------------------------------------
    # Sample-timing quality
    # ------------------------------------------------------------------

    def summarize_gaps(self) -> dict:
        """Summarize sample-timing quality: gap count/size, effective vs. nominal rate, and jitter."""
        return timing.summarize_gaps(self.time_stamps, self.nominal_srate)

    def locate_gaps(self) -> list[dict]:
        """The time range (seconds relative to this stream's own start) and size of each detected gap."""
        return timing.locate_gaps(self.time_stamps, self.nominal_srate)

    def gap_distribution(self, gap_concentration_threshold: float = 0.7) -> str:
        """Describe where detected gaps fall within the recording (concentrated or spread throughout)."""
        return timing.describe_gap_distribution(self.locate_gaps(), self.duration, gap_concentration_threshold)

    def classify_gaps(self, expected_duration: float | None = None) -> str:
        """Classify this stream's gap summary into a human-readable verdict."""
        return timing.classify_gaps(self.summarize_gaps(), self.locate_gaps(), expected_duration)

    def validate(self, expected_duration: float | None = None) -> dict:
        """Run the sample-timing checks on this stream and return its result row."""
        return timing.validate_stream(self, expected_duration)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, type={self.type!r}, hostname={self.hostname!r})"
