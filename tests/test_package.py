import importlib

import pytest

MODULES = [
    "audio",
    "clock",
    "constants",
    "filters",
    "file",
    "indexing",
    "plotting",
    "stream",
    "timing",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(f"xdf_tools.{module}")
