import inspect

from pyxdf import load_xdf

from xdf_tools.constants import _EXPECTED_WINSOR_THRESHOLD, WINSOR_THRESHOLD


def test_winsor_threshold_matches_pyxdf_default():
    """The drift-correction fit must scale by the same threshold pyxdf's own clock sync does.
    If pyxdf's default ever changes, this test fails (and a warning is logged on import)."""
    pyxdf_default = inspect.signature(load_xdf).parameters["winsor_threshold"].default
    assert WINSOR_THRESHOLD == pyxdf_default == _EXPECTED_WINSOR_THRESHOLD
