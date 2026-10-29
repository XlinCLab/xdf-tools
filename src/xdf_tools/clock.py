"""Clock synchronization: robust linear drift-correction fit, conversion of raw and stream-relative times onto
the shared LSL-synchronized time axis, and summaries of clock-offset measurements."""
import numpy as np
from pyxdf.pyxdf import _robust_fit

from xdf_tools.constants import WINSOR_THRESHOLD


class ClockResetError(ValueError):
    """Raised when a stream's clock-offset calibration data shows more than one segment,
    i.e. pyxdf detected a mid-recording clock discontinuity
    (e.g. if the recording device was restarted during recording).
    Fitting a single linear drift correction across a reset would be wrong,
    so this is treated as an error rather than handled."""


def fit_linear_drift(clock_times: np.ndarray, clock_values: np.ndarray) -> tuple[float, float]:
    """Fit the linear clock-offset correction `offset = intercept + slope * local_time` with pyxdf's
    Huber-robust regression (`pyxdf.pyxdf._robust_fit`) applied to clock-offset measurements.
    This reproduces the exact preprocessing that `pyxdf.pyxdf._clock_sync` does, in order to make the
    intercept and slope available (pyxdf applies the correction but does not expose it).

    Returns (intercept, slope), so that `synced = raw + intercept + slope * raw`.
    """
    clock_times = np.asarray(clock_times, dtype=np.float64)
    clock_values = np.asarray(clock_values, dtype=np.float64)
    if len(clock_times) < 2:
        raise ValueError("At least 2 clock-offset measurements are required to fit a drift correction.")
    design_matrix = np.column_stack([
        np.ones(len(clock_times)),
        clock_times / WINSOR_THRESHOLD,
    ])
    target = clock_values / WINSOR_THRESHOLD
    intercept, slope = _robust_fit(design_matrix, target)
    intercept *= WINSOR_THRESHOLD
    return float(intercept), float(slope)


def apply_clock_drift_correction(raw_time, drift_intercept: float, drift_slope: float):
    """Convert a raw (un-synchronized) timestamp or an array of raw timestamps
    onto the shared LSL-synchronized axis using a linear clock-drift correction
    with intercept and slope fit by `XDFStream.fit_drift_correction`:
    `synced = raw + drift_intercept + drift_slope * raw`

    Only valid for timestamps already on the same raw-clock domain as the stream's own
    `clock_times`/`clock_values` (e.g. a stream's own `time_stamps` array from a `synchronize_clocks=False` load).
    For regular-rate streams, pyxdf's default `dejitter_timestamps=True` regularizes raw
    per-sample timestamps before this correction is applied internally, so an XDF footer's
    `first_timestamp`/`last_timestamp` (which preserve the true, non-dejittered raw values) are NOT
    on this domain and must not be passed here.
    Use `convert_relative_time_to_synced` instead for times measured relative to a stream's own start
    (e.g. seconds into a WAV file exported from the stream).
    """
    return raw_time + drift_intercept + drift_slope * raw_time


def convert_relative_time_to_synced(relative_time, synced_start_time: float, drift_slope: float):
    """Convert a time measured in seconds elapsed since a stream's first sample (e.g. seconds
    into a WAV file exported from an audio stream) onto the shared LSL-synchronized axis:
    `synced = synced_start_time + relative_time * (1 + drift_slope)`

    Anchors directly on the stream's own already-synchronized start time rather than its raw start time,
    since for regular-rate streams the raw start time preserved in the XDF footer can differ
    from the raw start time pyxdf's clock-drift correction was actually fit against.
    """
    return synced_start_time + relative_time * (1 + drift_slope)


def summarize_clock_offsets(clock_values: np.ndarray) -> dict:
    """Summarize a stream's clock-offset calibration measurements: their number, mean,
    and the change from the first to the last measurement."""
    clock_values = np.asarray(clock_values, dtype=np.float64)
    n = len(clock_values)
    if n == 0:
        return {
            "n_measurements": 0,
            "mean_offset": 0.0,
            "drift": 0.0,
        }
    return {
        "n_measurements": n,
        "mean_offset": float(np.mean(clock_values)),
        "drift": float(clock_values[-1] - clock_values[0]) if n > 1 else 0.0,
    }
