"""XDFFile: the streams loaded from a single .xdf file."""
from __future__ import annotations

import logging

from pyxdf import load_xdf

from xdf_tools import timing
from xdf_tools.filters import (exclude_streams_by_name_substring,
                               filter_streams_by_type)
from xdf_tools.stream import XDFStream

logger = logging.getLogger(__name__)


class XDFFile:
    """The streams of an XDF file, loaded with pyxdf.

    By default, pyxdf's clock synchronization (every stream is moved onto the shared LSL clock using the clock
    offsets measured during recording) and timestamp dejittering (regular-rate streams get regular timestamps)
    are applied; pass explicit keyword arguments to override these defaults.
    """

    def __init__(self, path: str, **load_kwargs):
        self.path = path
        load_kwargs = {
            "synchronize_clocks": True,
            "dejitter_timestamps": True,
            **load_kwargs,
        }
        self.clocks_synced = bool(load_kwargs["synchronize_clocks"])
        raw_streams, self.header = load_xdf(path, **load_kwargs)
        self.streams = [XDFStream(stream, clocks_synced=self.clocks_synced) for stream in raw_streams]

    @staticmethod
    def _as_class(stream: XDFStream, stream_class: type[XDFStream]) -> XDFStream:
        return stream if stream_class is XDFStream else stream_class.from_stream(stream)

    def streams_by_type(
            self,
            stream_type: str | None = None,
            exclude_name_substring: str | None = None,
        ) -> list[XDFStream]:
        """Return streams matching a given type (case-insensitive; all types if None), optionally excluding
        streams whose name contains a given substring."""
        streams = filter_streams_by_type(self.streams, stream_type)
        return exclude_streams_by_name_substring(streams, exclude_name_substring)

    def stream_by_type(
            self,
            stream_type: str,
            exclude_name_substring: str | None = None,
            stream_class: type[XDFStream] = XDFStream,
        ) -> XDFStream:
        """Return the first stream matching a given type (case-insensitive), as an instance of `stream_class`."""
        matches = self.streams_by_type(stream_type, exclude_name_substring)
        if not matches:
            raise ValueError(f"No <{stream_type}> stream found in {self.path}")
        elif len(matches) > 1:
            logger.warning(f"Found {len(matches)} matching {stream_type} streams, returning only the first match.")
        return self._as_class(matches[0], stream_class)

    def streams_by_name(self, name: str) -> list[XDFStream]:
        """Return all streams with the specified name (exact match)."""
        return [stream for stream in self.streams if stream.name == name]

    def stream_by_name(self, name: str, stream_class: type[XDFStream] = XDFStream) -> XDFStream:
        """Return the first stream with the specified name, as an instance of `stream_class`
        (an `XDFStream` or a subclass of it)."""
        matches = self.streams_by_name(name)
        if not matches:
            raise ValueError(f"No <{name}> stream found in {self.path}")
        elif len(matches) > 1:
            logger.warning(f"Found {len(matches)} matching <{name}> streams, returning only the first match.")
        return self._as_class(matches[0], stream_class)

    def validate(self, stream_type: str | None = None, exclude_name_substring: str | None = None) -> dict:
        """Run sample-timing validation checks across this file's streams
        (optionally filtered by type and/or excluded by name substring).
        Returns a result dict describing each stream's validation outcome and an overall pass/fail status."""
        streams = self.streams_by_type(
            stream_type=stream_type,
            exclude_name_substring=exclude_name_substring,
        )
        return timing.validate_streams(self.path, streams)

    def __repr__(self):
        return f"XDFFile(path={self.path!r}, n_streams={len(self.streams)})"
