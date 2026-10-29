"""A minimal XDF writer for generating synthetic recordings in tests.

Writes the file header, one stream header, sample chunks, clock-offset chunks, and a stream footer per stream,
following the XDF 1.0 format: https://github.com/sccn/xdf/wiki/Specifications
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

import numpy as np

TAG_FILE_HEADER = 1
TAG_STREAM_HEADER = 2
TAG_SAMPLES = 3
TAG_CLOCK_OFFSET = 4
TAG_STREAM_FOOTER = 6

NUMERIC_FORMATS = {
    "float32": "<f4",
    "double64": "<f8",
    "int32": "<i4",
    "int16": "<i2",
    "int8": "i1",
}
SAMPLES_PER_CHUNK = 1000


@dataclass
class SyntheticStream:
    """A stream to write. `time_stamps` are on the stream's own (raw, local) clock.
    `data` is (n_samples, n_channels), or a list of per-sample lists of str for channel_format="string".
    The clock offsets (`clock_times`, `clock_values`) are what pyxdf adds to the raw timestamps to put them
    on the shared clock, so that synced = raw + offset."""
    name: str
    type: str
    hostname: str
    nominal_srate: float
    channel_format: str
    time_stamps: np.ndarray
    data: object
    labels: list[str] | None = None
    units: list[str] | None = None
    clock_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    clock_values: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def n_channels(self) -> int:
        if self.channel_format == "string":
            return len(self.data[0]) if len(self.data) else 1
        return np.asarray(self.data).shape[1]


def _varlen(value: int) -> bytes:
    if value < 2**8:
        return b"\x01" + struct.pack("<B", value)
    if value < 2**32:
        return b"\x04" + struct.pack("<I", value)
    return b"\x08" + struct.pack("<Q", value)


def _chunk(tag: int, content: bytes) -> bytes:
    return _varlen(len(content) + 2) + struct.pack("<H", tag) + content


def _stream_header_xml(stream: SyntheticStream) -> bytes:
    channels = ""
    if stream.labels is not None:
        units = stream.units or [""] * len(stream.labels)
        channels = "<desc><channels>" + "".join(
            f"<channel><label>{escape(label)}</label><unit>{escape(unit)}</unit></channel>"
            for label, unit in zip(stream.labels, units)
        ) + "</channels></desc>"
    xml = (
        "<?xml version=\"1.0\"?><info>"
        f"<name>{escape(stream.name)}</name><type>{escape(stream.type)}</type>"
        f"<channel_count>{stream.n_channels}</channel_count>"
        f"<nominal_srate>{stream.nominal_srate}</nominal_srate>"
        f"<channel_format>{stream.channel_format}</channel_format>"
        f"<source_id>{escape(stream.name)}@{escape(stream.hostname)}</source_id>"
        f"<hostname>{escape(stream.hostname)}</hostname>"
        f"{channels}</info>"
    )
    return xml.encode("utf-8")


def _samples_content(stream_id: int, stream: SyntheticStream, start: int, stop: int) -> bytes:
    n = stop - start
    header = struct.pack("<I", stream_id) + _varlen(n)
    time_stamps = np.asarray(stream.time_stamps, dtype=np.float64)[start:stop]
    if stream.channel_format == "string":
        body = bytearray()
        for ts, sample in zip(time_stamps, stream.data[start:stop]):
            body += b"\x08" + struct.pack("<d", ts)
            for value in sample:
                encoded = value.encode("utf-8")
                body += _varlen(len(encoded)) + encoded
        return header + bytes(body)
    value_dtype = np.dtype(NUMERIC_FORMATS[stream.channel_format])
    records = np.empty(n, dtype=np.dtype([("flag", "u1"), ("ts", "<f8"), ("values", value_dtype, (stream.n_channels,))]))
    records["flag"] = 8
    records["ts"] = time_stamps
    records["values"] = np.asarray(stream.data)[start:stop].astype(value_dtype)
    return header + records.tobytes()


def write_xdf(path, streams: list[SyntheticStream]) -> None:
    """Write streams (with stream ids 1, 2, ...) to an XDF file."""
    with open(path, "wb") as f:
        f.write(b"XDF:")
        f.write(_chunk(TAG_FILE_HEADER, b"<?xml version=\"1.0\"?><info><version>1.0</version></info>"))
        for stream_id, stream in enumerate(streams, start=1):
            f.write(_chunk(TAG_STREAM_HEADER, struct.pack("<I", stream_id) + _stream_header_xml(stream)))
            n_samples = len(stream.time_stamps)
            for start in range(0, n_samples, SAMPLES_PER_CHUNK):
                stop = min(start + SAMPLES_PER_CHUNK, n_samples)
                f.write(_chunk(TAG_SAMPLES, _samples_content(stream_id, stream, start, stop)))
            for collection_time, offset in zip(stream.clock_times, stream.clock_values):
                f.write(_chunk(TAG_CLOCK_OFFSET, struct.pack("<Idd", stream_id, collection_time, offset)))
        for stream_id, stream in enumerate(streams, start=1):
            time_stamps = np.asarray(stream.time_stamps, dtype=np.float64)
            footer = (
                "<?xml version=\"1.0\"?><info>"
                f"<first_timestamp>{float(time_stamps[0])!r}</first_timestamp>"
                f"<last_timestamp>{float(time_stamps[-1])!r}</last_timestamp>"
                f"<sample_count>{len(time_stamps)}</sample_count>"
                "</info>"
            ).encode("utf-8") if len(time_stamps) else b"<?xml version=\"1.0\"?><info></info>"
            f.write(_chunk(TAG_STREAM_FOOTER, struct.pack("<I", stream_id) + footer))
