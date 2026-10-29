"""Constants describing the structure of the stream dicts returned by `pyxdf.load_xdf`."""
import inspect
import logging

from pyxdf import load_xdf

logger = logging.getLogger(__name__)

# Top-level keys of a stream dict
STREAM_INFO = "info"
STREAM_TIME_SERIES = "time_series"
STREAM_TIME_STAMPS = "time_stamps"
STREAM_FOOTER = "footer"
# Raw clock-offset calibration measurements
STREAM_CLOCK_TIMES = "clock_times"
STREAM_CLOCK_VALUES = "clock_values"

# Keys of the XDF footer chunk's info (raw, un-synchronized boundary timestamps)
FOOTER_INFO_FIRST_TIMESTAMP = "first_timestamp"
FOOTER_INFO_LAST_TIMESTAMP = "last_timestamp"

# Keys within a stream's "info" metadata dict
INFO_NAME = "name"
INFO_TYPE = "type"
INFO_HOSTNAME = "hostname"
INFO_NOMINAL_SRATE = "nominal_srate"
INFO_CHANNEL_FORMAT = "channel_format"
INFO_DESC = "desc"
# Segment boundaries (index pairs into time_stamps) that pyxdf's clock-reset detection identified;
# only populated when loaded with synchronize_clocks=True
INFO_CLOCK_SEGMENTS = "clock_segments"

# Nested keys under info.desc.channels.channel[] describing each channel
INFO_CHANNELS = "channels"
INFO_CHANNEL = "channel"
INFO_LABEL = "label"
INFO_UNIT = "unit"

# pyxdf scales clock_times/clock_values by winsor_threshold before fitting drift correction
# Set the value directly from load_xdf's own default so this can never drift out of sync with pyxdf
WINSOR_THRESHOLD = inspect.signature(load_xdf).parameters["winsor_threshold"].default
_EXPECTED_WINSOR_THRESHOLD = 0.0001
if WINSOR_THRESHOLD != _EXPECTED_WINSOR_THRESHOLD:
    logger.warning(
        f"pyxdf's default value for `winsor_threshold` has changed: {_EXPECTED_WINSOR_THRESHOLD} -> {WINSOR_THRESHOLD}"
    )
