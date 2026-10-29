"""Stream filters, e.g. to disambiguate same-named streams recorded on different machines."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xdf_tools.stream import XDFStream


def filter_streams_by_type(streams: list[XDFStream], stream_type: str | None) -> list[XDFStream]:
    """Filter a list of streams down to those of a given type (case-insensitive); all streams if None."""
    if stream_type is None:
        return list(streams)
    return [stream for stream in streams if stream.type.lower() == str(stream_type).lower()]


def exclude_streams_by_name_substring(streams: list[XDFStream], substring: str | None) -> list[XDFStream]:
    """Drop streams whose name contains a given substring (case-insensitive), e.g. to exclude
    impedance-check streams which otherwise share the same type as the real data stream."""
    if substring is None:
        return list(streams)
    return [stream for stream in streams if substring.lower() not in stream.name.lower()]


def filter_streams_by_hostname(streams: list[XDFStream], hostname: str | list[str]) -> list[XDFStream]:
    """Filter a list of streams down to those recorded on one of one or more given host
    machines (case-insensitive). Used to disambiguate same-named streams from a recording that
    multiplexes streams of several recording rigs into one XDF file.
    A role may have more than one hostname if its recording rig was replaced partway
    through data collection."""
    hostnames = [hostname] if isinstance(hostname, str) else hostname
    allowed = {str(h).lower() for h in hostnames}
    return [stream for stream in streams if stream.hostname.lower() in allowed]
