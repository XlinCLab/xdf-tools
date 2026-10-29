import numpy as np

from xdf_tools.indexing import interval_index, nearest_index, nearest_indices

TIMESTAMPS = np.array([0.0, 1.0, 2.0, 5.0])


def test_nearest_indices():
    np.testing.assert_array_equal(nearest_indices([-3.0, 0.4, 0.6, 3.4, 3.6, 99.0], TIMESTAMPS), [0, 0, 1, 2, 3, 3])


def test_nearest_indices_ties_go_to_the_earlier_sample():
    np.testing.assert_array_equal(nearest_indices([0.5, 1.5, 3.5], TIMESTAMPS), [0, 1, 2])


def test_nearest_index():
    assert nearest_index(1.9, TIMESTAMPS) == 2
    assert isinstance(nearest_index(1.9, TIMESTAMPS), int)


def test_interval_index():
    starts = np.array([0.0, 10.0, 20.0])
    stops = np.array([5.0, 15.0, 25.0])
    times = [-1.0, 0.0, 5.0, 7.0, 10.0, 15.5, 22.0, 30.0]
    np.testing.assert_array_equal(interval_index(starts, stops, times), [-1, 0, 0, -1, 1, -1, 2, -1])
