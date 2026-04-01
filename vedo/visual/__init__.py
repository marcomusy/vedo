from __future__ import annotations

"""Public visual API."""

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    (
        "vedo.visual.runtime",
        [
            "CommonVisual",
            "PointsVisual",
            "VolumeVisual",
            "MeshVisual",
            "ImageVisual",
            "Actor2D",
            "Actor3DHelper",
            "LightKit",
        ],
    )
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
