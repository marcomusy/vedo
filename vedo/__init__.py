#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. include:: ../docs/documentation.md
"""
######################################################################## imports
import os
import sys
import warnings
import logging
import numpy as np
from numpy import sin, cos, sqrt, exp, log, dot, cross  # just because handy

from vtkmodules.vtkCommonCore import vtkVersion

#################################################
from vedo.version import _version as __version__

from vedo.settings import Settings
settings = Settings(level=0)

from vedo.utils import *
from vedo.colors import *
from vedo.shapes import *
from vedo.io import *
from vedo.base import *
from vedo.ugrid import *
from vedo.assembly import *
from vedo.pointcloud import *
from vedo.mesh import *
from vedo.picture import *
from vedo.volume import *
from vedo.tetmesh import *
from vedo.addons import *
from vedo.plotter import *

from vedo import applications

try:
    import platform
    sys_platform = platform.system()
except (ModuleNotFoundError, AttributeError) as e:
    sys_platform = ""

#######################################################################################
__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy"
__email__      = "marco.musy@embl.es"
__website__    = "https://github.com/marcomusy/vedo"


##### To generate documentation #######################################################
# cd Projects/vedo
# pip uninstall vedo
# Uncomment the lines below __pdoc__[] ...
# pdoc3 --html . --force -c lunr_search="{'fuzziness': 0, 'index_docstrings': True}"
# chmod 755 html/ -R
# mount_staging
# rm ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/html
# mv html/ ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/autodocs/
########################################## pdoc excludes:
# __pdoc__ = {}
# __pdoc__['backends'] = False
# __pdoc__['cli'] = False
# __pdoc__['cmaps'] = False
# __pdoc__['version'] = False
# __pdoc__['vtkclasses'] = False

# __pdoc__['colors.getColor'] = False
# __pdoc__['colors.colorMap'] = False
# __pdoc__['colors.buildLUT'] = False

# __pdoc__['base.Base3DProp.rotateX'] = False
# __pdoc__['base.Base3DProp.rotateY'] = False
# __pdoc__['base.Base3DProp.rotateZ'] = False
# __pdoc__['base.Base3DProp.applyTransform'] = False
# __pdoc__['base.Base3DProp.diagonalSize'] = False
# __pdoc__['base.BaseActor.N'] = False
# __pdoc__['base.BaseActor.NPoints'] = False
# __pdoc__['base.BaseActor.NCells'] = False
# __pdoc__['base.BaseActor.cellCenters'] = False
# __pdoc__['base.BaseActor.addScalarBar'] = False
# __pdoc__['base.BaseActor.addScalarBar3D'] = False
# __pdoc__['base.BaseGrid.cutWithPlane'] = False

# __pdoc__['mesh.Mesh.computeNormals'] = False
# __pdoc__['mesh.Mesh.backColor'] = False
# __pdoc__['mesh.Mesh.lineWidth'] = False
# __pdoc__['mesh.Mesh.addCurvatureScalars'] = False
# __pdoc__['mesh.Mesh.addShadow'] = False
# __pdoc__['mesh.Mesh.fillHoles'] = False
# __pdoc__['mesh.Mesh.intersectWithLine'] = False
# __pdoc__['mesh.Mesh.followCamera'] = False
# __pdoc__['mesh.Mesh.extractLargestRegion'] = False
# __pdoc__['mesh.Mesh.intersectWith'] = False
# __pdoc__['mesh.Mesh.signedDistance'] = False

# __pdoc__['plotter.Plotter.getMeshes'] = False
# __pdoc__['plotter.Plotter.resetCamera'] = False
# __pdoc__['plotter.Plotter.addSlider2D'] = False
# __pdoc__['plotter.Plotter.addButton'] = False
# __pdoc__['plotter.Plotter.addSplineTool'] = False
# __pdoc__['plotter.Plotter.addCallback'] = False

# __pdoc__['settings.Settings.allowInteraction'] = False
# __pdoc__['settings.Settings.alphaBitPlanes'] = False
# __pdoc__['settings.Settings.defaultFont'] = False
# __pdoc__['settings.Settings.enableDefaultKeyboardCallbacks'] = False
# __pdoc__['settings.Settings.enableDefaultMouseCallbacks'] = False
# __pdoc__['settings.Settings.enablePrintColor'] = False
# __pdoc__['settings.Settings.hackCallScreenSize'] = False
# __pdoc__['settings.Settings.hiddenLineRemoval'] = False
# __pdoc__['settings.Settings.immediateRendering'] = False
# __pdoc__['settings.Settings.interpolateScalarsBeforeMapping'] = False
# __pdoc__['settings.Settings.lightFollowsCamera'] = False
# __pdoc__['settings.Settings.lineSmoothing'] = False
# __pdoc__['settings.Settings.maxNumberOfPeels'] = False
# __pdoc__['settings.Settings.multiSamples'] = False
# __pdoc__['settings.Settings.occlusionRatio'] = False
# __pdoc__['settings.Settings.pointSmoothing'] = False
# __pdoc__['settings.Settings.polygonOffsetFactor'] = False
# __pdoc__['settings.Settings.polygonOffsetUnits'] = False
# __pdoc__['settings.Settings.polygonSmoothing'] = False
# __pdoc__['settings.Settings.preserveDepthBuffer'] = False
# __pdoc__['settings.Settings.rememberLastFigureFormat'] = False
# __pdoc__['settings.Settings.renderLinesAsTubes'] = False
# __pdoc__['settings.Settings.rendererFrameAlpha'] = False
# __pdoc__['settings.Settings.rendererFrameColor'] = False
# __pdoc__['settings.Settings.rendererFramePadding'] = False
# __pdoc__['settings.Settings.rendererFrameWidth'] = False
# __pdoc__['settings.Settings.screenshotTransparentBackground'] = False
# __pdoc__['settings.Settings.screeshotLargeImage'] = False
# __pdoc__['settings.Settings.screeshotScale'] = False
# __pdoc__['settings.Settings.tiffOrientationType'] = False
# __pdoc__['settings.Settings.twoSidedLighting'] = False
# __pdoc__['settings.Settings.useDepthPeeling'] = False
# __pdoc__['settings.Settings.useFXAA'] = False
# __pdoc__['settings.Settings.useParallelProjection'] = False
# __pdoc__['settings.Settings.usePolygonOffset'] = False
# __pdoc__['settings.Settings.visibleGridEdges'] = False
# __pdoc__['settings.Settings.windowSplittingPosition'] = False

# __pdoc__['pointcloud.pcaEllipsoid'] = False
# __pdoc__['pointcloud.Points.cellIndividualColors'] = False
# __pdoc__['pointcloud.Points.centerOfMass'] = False
# __pdoc__['pointcloud.Points.alignTo'] = False
# __pdoc__['pointcloud.Points.applyTransform'] = False
# __pdoc__['pointcloud.Points.interpolateDataFrom'] = False
# __pdoc__['pointcloud.Points.closestPoint'] = False
# __pdoc__['pointcloud.Points.cutWithPlane'] = False
# __pdoc__['pointcloud.Points.cutWithMesh'] = False
# __pdoc__['pointcloud.Points.reconstructSurface'] = False
# __pdoc__['pointcloud.Points.signedDistance'] = False
# __pdoc__['pointcloud.Points.distanceTo'] = False
# __pdoc__['pointcloud.Points.pointSize'] = False
# __pdoc__['pointcloud.Points.vignette'] = False
# __pdoc__['pointcloud.Points.cell_individual_colors'] = False

# __pdoc__['volume.BaseVolume.scalarRange'] = False
# __pdoc__['volume.BaseVolume.slicePlane'] = False

##################################################################################
########################################################################## GLOBALS
vtk_version = (
    int(vtkVersion().GetVTKMajorVersion()),
    int(vtkVersion().GetVTKMinorVersion()),
    int(vtkVersion().GetVTKBuildVersion()),
)

# if vtk_version[0] >= 9:
#     if "Windows" in sys_platform or "Linux" in sys_platform:
#         settings.use_depth_peeling = True

installdir = os.path.dirname(__file__)
dataurl = "https://vedo.embl.es/examples/data/"

plotter_instance = None
notebook_plotter = None
notebook_backend = None

## fonts
fonts_path = os.path.join(installdir, "fonts/")

## a fatal error occurs when compiling to exe,
## developer needs to copy the fonts folder to the same location as the exe file to solve this problem
if not os.path.exists(fonts_path):
    fonts_path = "fonts/"

fonts = [_f.split(".")[0] for _f in os.listdir(fonts_path) if '.npz' not in _f]
fonts = list(sorted(fonts))

# pyplot module to remember last figure format
last_figure = None


######################################################################### logging
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
        return formatter.format(record)

logger = logging.getLogger("vedo")

_chsh = logging.StreamHandler()
_chsh.flush = sys.stdout.flush
_chsh.setLevel(logging.DEBUG)
_chsh.setFormatter(_LoggingCustomFormatter())
logger.addHandler(_chsh)
logger.setLevel(logging.INFO)

################################################# silence annoying messages
warnings.simplefilter(action="ignore", category=FutureWarning)
try:
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
except AttributeError:
    pass


