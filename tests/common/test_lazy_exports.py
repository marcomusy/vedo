#!/usr/bin/env python3
from __future__ import annotations

"""Guardrails for lazy export registration across vedo facades."""

import importlib


MODULES = [
    "vedo",
    "vedo.core",
    "vedo.shapes",
    "vedo.file_io",
    "vedo.mesh",
    "vedo.grids",
    "vedo.volume",
    "vedo.pointcloud",
    "vedo.addons",
    "vedo.plotter",
    "vedo.visual",
    "vedo.applications",
    "vedo.pyplot",
    "vedo.external",
]

# Some package-level names are intentionally available for internal compatibility
# even though they are not public __all__ exports.
COMPAT_NAMES = {
    "vedo.addons": [
        "BaseCutter",
        "SliderWidget",
        "PointCloudWidget",
        "compute_visible_bounds",
    ],
    "vedo.visual": [
        "Actor3DHelper",
    ],
    "vedo.shapes": [
        "_reps",
    ],
    "vedo.grids.image": [
        "_get_img",
    ],
}


def _assert_names_resolve(module_name: str, names: list[str] | tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)
    module_dir = dir(module)
    for name in names:
        assert name in module_dir, f"{module_name}.{name} missing from dir()"
        getattr(module, name)


def test_lazy_exports_resolve() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        exports = list(getattr(module, "__all__", ()))
        assert exports, f"{module_name} has no __all__ exports"
        _assert_names_resolve(module_name, exports)

    for module_name, names in COMPAT_NAMES.items():
        _assert_names_resolve(module_name, names)
