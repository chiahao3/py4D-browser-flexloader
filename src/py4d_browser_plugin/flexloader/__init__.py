__version__ = "0.0.3"

try:
    from .flexloader import FlexLoaderPlugin
except ModuleNotFoundError as exc:
    if exc.name != "PyQt5":
        raise
    __all__ = []
else:
    __all__ = ["FlexLoaderPlugin"]
