#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
#
##### To generate documentation
# cd ~/Projects/vedo/docs/pdoc
# ./build_html.py
###############################
"""
.. include:: ../docs/documentation.md
"""

######################################################################## imports
import os
import sys
import logging
from importlib.metadata import PackageNotFoundError, version as pkg_version
import numpy as np
from numpy import sin, cos, sqrt, exp, log, dot, cross  # just because handy

from vedo.lazy_imports import build_attr_map, dir_lazy, getattr_lazy

try:
    from vtkmodules.vtkCommonCore import vtkVersion
except ModuleNotFoundError:
    print("Cannot find VTK installation. Please install it with:")
    print("pip install vtk")
    sys.exit(1)

#################################################
try:
    __version__ = pkg_version("vedo")
except PackageNotFoundError:
    __version__ = "2026.6.1.dev02"  # fallback version if package metadata is not available

from vedo.plotter import session as _session

from vedo.settings import Settings
settings = Settings()

try:
    import platform
    sys_platform = platform.system()
except (ModuleNotFoundError, AttributeError):
    sys_platform = ""

######################################################################### GLOBALS
__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy"
__email__      = "marco.musy@embl.es"
__website__    = "https://github.com/marcomusy/vedo"


##########################################################################
vtk_version = (
    int(vtkVersion().GetVTKMajorVersion()),
    int(vtkVersion().GetVTKMinorVersion()),
    int(vtkVersion().GetVTKBuildVersion()),
)

installdir = os.path.dirname(__file__)
dataurl = "https://vedo.embl.es/examples/data/"

plotter_instance = None
notebook_plotter = None
notebook_backend = None

## fonts
_fonts_dir_candidates = [
    os.path.join(installdir, "fonts"),
    os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "fonts"),
    os.path.join(os.getcwd(), "fonts"),
]
fonts_path = ""
for _candidate in _fonts_dir_candidates:
    if os.path.isdir(_candidate):
        fonts_path = _candidate
        break

if fonts_path:
    # Keep a unique, sorted list while supporting both source and packaged layouts.
    fonts = sorted(
        {
            os.path.splitext(_f)[0]
            for _f in os.listdir(fonts_path)
            if _f.endswith((".ttf", ".npz"))
        }
    )
else:
    fonts = []

# pyplot module to remember last figure format
last_figure = None

_LAZY_EXPORT_MAP, _LAZY_EXPORTS = build_attr_map(
    ("vedo.colors", ["printc", "printd", "get_color", "get_color_name", "color_map", "build_palette", "build_lut"]),
    (
        "vedo.core.transformations",
        [
            "Quaternion",
            "LinearTransform",
            "NonLinearTransform",
            "TransformInterpolator",
            "spher2cart",
            "cart2spher",
            "cart2cyl",
            "cyl2cart",
            "cyl2spher",
            "spher2cyl",
            "cart2pol",
            "pol2cart",
        ],
    ),
    (
        "vedo.utils",
        [
            "OperationNode",
            "ProgressBar",
            "progressbar",
            "Minimizer",
            "compute_hessian",
            "geometry",
            "is_sequence",
            "lin_interpolate",
            "vector",
            "mag",
            "mag2",
            "versor",
            "precision",
            "round_to_digit",
            "point_in_triangle",
            "point_line_distance",
            "otsu_threshold",
            "closest",
            "grep",
            "make_bands",
            "pack_spheres",
            "humansort",
            "print_histogram",
            "print_inheritance_tree",
            "camera_from_quaternion",
            "camera_from_neuroglancer",
            "camera_from_dict",
            "camera_to_dict",
            "oriented_camera",
            "vtk2numpy",
            "numpy2vtk",
            "get_uv",
            "andrews_curves",
        ],
    ),
    ("vedo.core", ["DataArrayHelper", "CommonAlgorithms", "PointAlgorithms", "VolumeAlgorithms"]),
    (
        "vedo.shapes",
        [
            "Marker",
            "Line",
            "DashedLine",
            "RoundedLine",
            "Tube",
            "Tubes",
            "ThickTube",
            "Lines",
            "Spline",
            "KSpline",
            "CSpline",
            "Bezier",
            "Brace",
            "NormalLines",
            "Ribbon",
            "Arrow",
            "Arrows",
            "Arrow2D",
            "Arrows2D",
            "FlatArrow",
            "Polygon",
            "Triangle",
            "Rectangle",
            "Disc",
            "Circle",
            "GeoCircle",
            "Arc",
            "Star",
            "Star3D",
            "Cross3D",
            "IcoSphere",
            "Sphere",
            "Spheres",
            "Earth",
            "Ellipsoid",
            "Grid",
            "TessellatedBox",
            "Plane",
            "Box",
            "Cube",
            "Spring",
            "Cylinder",
            "Cone",
            "Pyramid",
            "Torus",
            "Paraboloid",
            "Hyperboloid",
            "Text2D",
            "Text3D",
            "Latex",
            "Glyph",
            "Tensors",
            "ParametricShape",
            "ConvexHull",
            "VedoLogo",
        ],
    ),
    (
        "vedo.file_io",
        [
            "load",
            "read",
            "download",
            "gunzip",
            "loadStructuredPoints",
            "loadStructuredGrid",
            "write",
            "save",
            "export_window",
            "import_window",
            "load_obj",
            "screenshot",
            "ask",
            "Video",
            "file_info",
            "load3DS",
            "loadOFF",
            "loadSTEP",
            "loadGeoJSON",
            "loadPVD",
            "loadNeutral",
            "loadGmesh",
            "loadPCD",
            "from_numpy",
            "loadImageData",
            "to_numpy",
        ],
    ),
    ("vedo.assembly", ["Group", "Assembly"]),
    (
        "vedo.pointcloud",
        [
            "Points",
            "Point",
            "merge",
            "fit_line",
            "fit_circle",
            "fit_plane",
            "fit_sphere",
            "pca_ellipse",
            "pca_ellipsoid",
            "project_point_on_variety",
            "procrustes_alignment",
        ],
    ),
    ("vedo.mesh", ["Mesh"]),
    ("vedo.grids.image", ["Image"]),
    ("vedo.volume", ["Volume"]),
    ("vedo.grids", ["UnstructuredGrid", "TetMesh", "RectilinearGrid", "StructuredGrid", "ExplicitStructuredGrid"]),
    (
        "vedo.addons",
        [
            "ScalarBar",
            "ScalarBar3D",
            "Slider2D",
            "Slider3D",
            "Icon",
            "LegendBox",
            "Light",
            "Axes",
            "RendererFrame",
            "Ruler2D",
            "Ruler3D",
            "RulerAxes",
            "DistanceTool",
            "SplineTool",
            "DrawingWidget",
            "Goniometer",
            "Button",
            "ButtonWidget",
            "Flagpost",
            "ProgressBarWidget",
            "BoxCutter",
            "PlaneCutter",
            "SphereCutter",
        ],
    ),
    ("vedo.plotter", ["Plotter", "show", "close"]),
    ("vedo.visual", ["CommonVisual", "PointsVisual", "VolumeVisual", "MeshVisual", "ImageVisual", "Actor2D", "LightKit"]),
)

_LAZY_MODULES = {
    "vtkclasses": "vedo.vtkclasses",
    "colors": "vedo.colors",
    "utils": "vedo.utils",
    "transformations": "vedo.core.transformations",
    "core": "vedo.core",
    "shapes": "vedo.shapes",
    "file_io": "vedo.file_io",
    "assembly": "vedo.assembly",
    "pointcloud": "vedo.pointcloud",
    "mesh": "vedo.mesh",
    "grids": "vedo.grids",
    "volume": "vedo.volume",
    "addons": "vedo.addons",
    "plotter": "vedo.plotter",
    "visual": "vedo.visual",
    "applications": "vedo.applications",
    "external": "vedo.external",
    "pyplot": "vedo.pyplot",
    "backends": "vedo.backends",
}

__all__ = [
    "np",
    "sin",
    "cos",
    "sqrt",
    "exp",
    "log",
    "dot",
    "cross",
    "settings",
    "installdir",
    "dataurl",
    "fonts_path",
    "fonts",
    "plotter_instance",
    "notebook_plotter",
    "notebook_backend",
    "last_figure",
    "sys_platform",
    "vtk_version",
    "logger",
    "current_plotter",
    "set_current_plotter",
    "current_notebook_plotter",
    "set_current_notebook_plotter",
    "current_notebook_backend",
    "set_current_notebook_backend",
    "current_last_figure",
    "set_last_figure",
    *_LAZY_EXPORTS,
]


######################################################################### LOGGING
class _LoggingCustomFormatter(logging.Formatter):

    logformat = "[vedo.%(filename)s:%(lineno)d] %(levelname)s: %(message)s"

    white = "\x1b[1m"
    grey = "\x1b[2m\x1b[1m\x1b[38;20m"
    yellow = "\x1b[1m\x1b[33;20m"
    red = "\x1b[1m\x1b[31;20m"
    inv_red = "\x1b[7m\x1b[1m\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey  + logformat + reset,
        logging.INFO: white + logformat + reset,
        logging.WARNING: yellow + logformat + reset,
        logging.ERROR: red + logformat + reset,
        logging.CRITICAL: inv_red + logformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record).replace(".py", "")

logger = logging.getLogger("vedo")

_log_stream = sys.stdout if sys.stdout is not None else sys.__stdout__
if _log_stream is None:
    _log_stream = open(os.devnull, "w")
_chsh = logging.StreamHandler(_log_stream)
_chsh.setLevel(logging.DEBUG)
_chsh.setFormatter(_LoggingCustomFormatter())
# Avoid duplicate handlers when vedo is re-imported/reloaded.
if not any(
    isinstance(h, logging.StreamHandler)
    and getattr(h, "_vedo_default_handler", False)
    for h in logger.handlers
):
    _chsh._vedo_default_handler = True  # type: ignore[attr-defined]
    logger.addHandler(_chsh)
logger.setLevel(logging.INFO)
logger.propagate = False


def current_plotter():
    """Return the active plotter instance for the current runtime session."""
    return _session.get_plotter(plotter_instance)


def set_current_plotter(plotter):
    """Set the active plotter instance for the current runtime session."""
    global plotter_instance
    plotter_instance = plotter
    _session.set_plotter(plotter)
    return plotter


def current_notebook_plotter():
    """Return the active notebook plotter object for the current runtime session."""
    return _session.get_notebook_plotter(notebook_plotter)


def set_current_notebook_plotter(plotter):
    """Set the active notebook plotter object for the current runtime session."""
    global notebook_plotter
    notebook_plotter = plotter
    _session.set_notebook_plotter(plotter)
    return plotter


def current_notebook_backend():
    """Return the active notebook backend for the current runtime session."""
    return _session.get_notebook_backend(notebook_backend)


def set_current_notebook_backend(backend):
    """Set the active notebook backend for the current runtime session."""
    global notebook_backend
    notebook_backend = backend
    _session.set_notebook_backend(backend)
    return backend


def current_last_figure():
    """Return the last pyplot figure format remembered in this runtime session."""
    return _session.get_last_figure(last_figure)


def set_last_figure(figure):
    """Set the last pyplot figure format remembered in this runtime session."""
    global last_figure
    last_figure = figure
    _session.set_last_figure(figure)
    return figure


def __getattr__(name):
    """Lazy-load public API symbols and selected modules."""
    return getattr_lazy(
        __name__,
        globals(),
        name,
        attr_map=_LAZY_EXPORT_MAP,
        module_map=_LAZY_MODULES,
    )


def __dir__():
    return dir_lazy(globals(), attr_map=_LAZY_EXPORT_MAP, module_map=_LAZY_MODULES)
