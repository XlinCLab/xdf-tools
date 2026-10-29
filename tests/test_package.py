import importlib

import pytest

# Modules available with the core install, and those needing an optional extra
CORE_MODULES = [
    "clock",
    "constants",
    "file",
    "filters",
    "indexing",
    "stream",
    "timing",
]
EXTRA_MODULES = {
    "audio": "scipy",
    "plotting": "pyqtgraph",
}


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
    importlib.import_module(f"xdf_tools.{module}")


@pytest.mark.parametrize("module", EXTRA_MODULES)
def test_extra_module_imports(module):
    pytest.importorskip(EXTRA_MODULES[module])
    importlib.import_module(f"xdf_tools.{module}")
