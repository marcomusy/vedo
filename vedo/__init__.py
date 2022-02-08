#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific analysis and visualization of 3D objects and point clouds based on VTK and numpy.
"""
__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vedo"

#######################################################################################
import os
import sys
import vtk
import warnings
import logging
import numpy as np
from deprecated import deprecated
from numpy import sin, cos, sqrt, exp, log, dot, cross  # just because are useful

#######################################################################################
from vedo.version import _version as __version__
from vedo.utils import *
from vedo.settings import _setts as settings
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
from vedo.shapes import *
from vedo.addons import *
from vedo.plotter import *

# HACK: need to uncomment this to generate html documentation
#from vedo.dolfin import _inputsort
#import vedo.base as base
#import vedo.docs as docs  # needed by spyder console, otherwise complains


##################################################################################
########################################################################## GLOBALS
vtk_version = [ int(vtk.vtkVersion().GetVTKMajorVersion()),
                int(vtk.vtkVersion().GetVTKMinorVersion()),
                int(vtk.vtkVersion().GetVTKBuildVersion()),
]
try:
    import platform
    sys_platform = platform.system()
except:
    sys_platform = ""

if vtk_version[0] >= 9:
    if "Windows" in sys_platform:
        settings.useDepthPeeling = True

warnings.simplefilter(action="ignore", category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

logging.basicConfig(
    format="[%(filename)s:%(lineno)d] %(levelname)s: %(message)s", level=logging.DEBUG,
)
logger = logging.getLogger("vedo")
#_lsthdr = logging.StreamHandler(sys.stdout)
#_lsthdr.flush = sys.stdout.flush
#logger.addHandler(_lsthdr)


################################################################################
installdir = os.path.dirname(__file__)
dataurl    = "https://vedo.embl.es/examples/data/"

plotter_instance = None
notebook_plotter = None
notebookBackend  = None

interactorStyle = None  # internal use only

## textures
textures_path = os.path.join(installdir, "textures/")
textures = []
for f in os.listdir(textures_path):
    tfn = f.split(".")[0]
    if 'earth' in tfn: continue
    textures.append(tfn)

## fonts
fonts_path = os.path.join(installdir, "fonts/")
fonts = []
for f in os.listdir(fonts_path):
    if '.npz' in f: continue
    fonts.append(f.split(".")[0])
fonts = list(sorted(fonts))

################################################################################## deprecated
@deprecated(reason="Please use Plotter(backend='...')")
def embedWindow(backend='ipyvtk', verbose=True):
    """Use this function to control whether the rendering window is inside
    the jupyter notebook or as an independent external window"""
    global notebook_plotter, notebookBackend

    if not backend:
        notebookBackend = None
        notebook_plotter = None
        return ####################

    else:

        if any(['SPYDER' in name for name in os.environ]):
            notebookBackend = None
            notebook_plotter = None
            return

        try:
            get_ipython()
        except NameError:
            notebookBackend = None
            notebook_plotter = None
            return

    backend = backend.lower()
    notebookBackend = backend

    if backend=='k3d':
        try:
            import k3d
            if k3d._version.version_info != (2, 7, 4):
                print('Warning: only k3d version 2.7.4 is currently supported')
                # print('> pip install k3d==2.7.4')

        except ModuleNotFoundError:
            notebookBackend = None
            if verbose:
                print('embedWindow(verbose=True): could not load k3d module, try:')
                print('> pip install k3d==2.7.4')

    elif 'ipygany' in backend: # ipygany
        try:
            import ipygany
        except ModuleNotFoundError:
            notebookBackend = None
            if verbose:
                print('embedWindow(verbose=True): could not load ipygany module, try:')
                print('> pip install ipygany')

    elif 'itk' in backend: # itkwidgets
        try:
            import itkwidgets
        except ModuleNotFoundError:
            notebookBackend = None
            if verbose:
                print('embedWindow(verbose=True): could not load itkwidgets module, try:')
                print('> pip install itkwidgets')

    elif backend.lower() == '2d':
        pass

    elif backend =='panel':
        try:
            import panel
            panel.extension('vtk')
        except:
            if verbose:
                print('embedWindow(verbose=True): could not load panel try:')
                print('> pip install panel')

    elif 'ipyvtk' in backend:
        try:
            from ipyvtklink.viewer import ViewInteractiveWidget
        except ModuleNotFoundError:
            if verbose:
                print('embedWindow(verbose=True): could not load ipyvtklink try:')
                print('> pip install ipyvtklink')

    else:
        print("Unknown backend", backend)
        raise RuntimeError()

