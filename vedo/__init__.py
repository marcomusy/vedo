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
import numpy as np
from numpy import sin, cos, sqrt, exp, log, dot, cross  # just because handy

from vtkmodules.vtkCommonCore import vtkVersion

#################################################
from vedo.version import _version as __version__

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

from vedo import applications
from vedo import interactor_modes

try:
    import platform
    sys_platform = platform.system()
except (ModuleNotFoundError, AttributeError) as e:
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

    logformat = "[vedo.%(filename)s] %(levelname)s: %(message)s"

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
_chsh.flush = sys.stdout.flush
_chsh.setLevel(logging.DEBUG)
_chsh.setFormatter(_LoggingCustomFormatter())
logger.addHandler(_chsh)
logger.setLevel(logging.INFO)

