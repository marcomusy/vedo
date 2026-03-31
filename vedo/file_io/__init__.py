from __future__ import annotations
"""
Submodule to read/write meshes and other objects in different formats,
and other I/O functionalities.
"""

__docformat__ = "google"

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

_LAZY_EXPORT_MAP, __all__ = build_attr_map(
    (
        "vedo.file_io.loaders",
        [
            "from_numpy",
            "load",
            "load3DS",
            "loadGeoJSON",
            "loadGmesh",
            "loadImageData",
            "loadNeutral",
            "loadOFF",
            "loadPCD",
            "loadPVD",
            "loadSTEP",
            "loadStructuredGrid",
            "loadStructuredPoints",
            "load_obj",
        ],
    ),
    ("vedo.file_io.network", ["download", "file_info", "gunzip"]),
    ("vedo.file_io.scene", ["export_window", "import_window", "screenshot", "to_numpy"]),
    ("vedo.file_io.terminal", ["ask"]),
    ("vedo.file_io.video", ["Video"]),
    ("vedo.file_io.writers", ["read", "save", "write"]),
)


def __getattr__(name):
    return getattr_lazy(__name__, globals(), name, attr_map=_LAZY_EXPORT_MAP)


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP)
