"""Time-to-sample lookup helpers."""
import numpy as np


def nearest_indices(target_times: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """For each target time, the index of the closest (sorted) timestamp; ties go to the earlier one."""
    target_times = np.asarray(target_times)
    idx = np.searchsorted(timestamps, target_times)
    after = np.clip(idx, 0, len(timestamps) - 1)
    before = np.clip(idx - 1, 0, len(timestamps) - 1)
    closer_before = np.abs(timestamps[before] - target_times) <= np.abs(timestamps[after] - target_times)
    return np.where(closer_before, before, after)


def nearest_index(target_time: float, timestamps: np.ndarray) -> int:
    """Index of the closest (sorted) timestamp to a single time."""
    return int(nearest_indices(np.array([target_time]), timestamps)[0])


def interval_index(starts: np.ndarray, stops: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the interval [start, stop] that contains each time, -1 where none does.
    The intervals must be sorted by start and must not overlap.
    """
    times = np.asarray(times)
    index = np.searchsorted(starts, times, side="right") - 1
    inside = (index >= 0) & (times <= np.asarray(stops)[np.clip(index, 0, None)])
    return np.where(inside, index, -1)
