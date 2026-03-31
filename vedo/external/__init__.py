from __future__ import annotations
"""Helpers integrating optional external libraries."""

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    (
        "vedo.external.conversions",
        [
            "vedo2trimesh",
            "trimesh2vedo",
            "vedo2meshlab",
            "meshlab2vedo",
            "vedo2open3d",
            "open3d2vedo",
            "vedo2madcad",
            "madcad2vedo",
        ],
    )
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
