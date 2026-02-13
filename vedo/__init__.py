#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import importlib
import numpy as np
from numpy import sin, cos, sqrt, exp, log, dot, cross  # just because handy

try:
    from vtkmodules.vtkCommonCore import vtkVersion
except ModuleNotFoundError:
    print("Cannot find VTK installation. Please install it with:")
    print("pip install vtk")
    sys.exit(1)

#################################################
from vedo.version import _version as __version__
from vedo import session as _session

from vedo.settings import Settings
settings = Settings()

from vedo.colors import *
from vedo.transformations import *
from vedo.utils import *
from vedo.core import *
from vedo.shapes import *
from vedo.file_io import *
from vedo.assembly import *
from vedo.pointcloud import *
from vedo.mesh import *
from vedo.image import *
from vedo.volume import *
from vedo.grids import *
from vedo.addons import *
from vedo.plotter import *
from vedo.visual import *

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
fonts_path = os.path.join(installdir, "fonts/")

# Note:
# a fatal error occurs when compiling to exe,
# developer needs to copy the fonts folder to the same location as the exe file
# to solve this problem
if not os.path.exists(fonts_path):
    fonts_path = "fonts/"

fonts = [_f.split(".")[0] for _f in os.listdir(fonts_path) if '.npz' not in _f]
fonts = list(sorted(fonts))

# pyplot module to remember last figure format
last_figure = None


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

_chsh = logging.StreamHandler()
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
_chsh.flush = sys.stdout.flush
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
    """Lazy-load selected heavy optional modules while preserving public API."""
    if name in {"applications", "interactor_modes"}:
        module = importlib.import_module(f"vedo.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
