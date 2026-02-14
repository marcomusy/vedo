"""Public plotter API."""

from importlib import import_module

__all__ = ["Plotter", "show", "close"]


def __getattr__(name):
    if name in {"Plotter", "show", "close"}:
        module = import_module("vedo.plotter.runtime")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
