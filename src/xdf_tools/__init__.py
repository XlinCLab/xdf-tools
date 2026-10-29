from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xdf-tools")
except PackageNotFoundError:  # not installed (e.g. imported straight from a source checkout)
    __version__ = "0.0.0"
