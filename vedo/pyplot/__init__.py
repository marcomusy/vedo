from __future__ import annotations
"""Advanced plotting functionalities."""

from vedo._lazy import build_attr_map, dir_lazy, getattr_lazy

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    ("vedo.pyplot.figure", ["Figure"]),
    ("vedo.pyplot.charts", ["Histogram1D", "Histogram2D", "PlotBars", "PlotXY"]),
    ("vedo.pyplot.functions", ["plot", "histogram", "fit", "streamplot"]),
    ("vedo.pyplot.stats", ["pie_chart", "violin", "whisker", "matrix", "CornerPlot", "CornerHistogram"]),
    ("vedo.pyplot.graph", ["DirectedGraph"]),
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
