#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. image:: https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png

A python module for scientific analysis of 3D objects and
point clouds based on [VTK](https://www.vtk.org/) and [numpy](http://www.numpy.org/).

# Documentation
Use this page to search and inspect `vedo` sub-modules, methods and functions.
These documentation pages are automatically generated
by [pdoc](https://pdoc3.github.io/pdoc/) from the source files.

## Quick start
```bash
pip install vedo
```
Then
```python
from vedo import Cone
cone = Cone()      # Create a simple cone
cone.show(axes=1)  # Show it (with axes)
```
.. image:: https://vedo.embl.es/images/feats/cone.png


## Command Line Interface
The library comes with a convenient Command Line Interface. Type for example in your terminal:

```bash
vedo --help
vedo https://vedo.embl.es/examples/data/panther.stl.gz
```
.. image:: https://vedo.embl.es/images/feats/vedo_cli_panther.png

Pressing `h` will then show a number of options to interact with your 3D scene:
```
 ==========================================================
| Press: i     print info about selected object            |
|        I     print the RGB color under the mouse         |
|        <-->  use arrows to reduce/increase opacity       |
|        w/s   toggle wireframe/surface style              |
|        p/P   change point size of vertices               |
|        l     toggle edges visibility                     |
|        x     toggle mesh visibility                      |
|        X     invoke a cutter widget tool                 |
|        1-3   change mesh color                           |
|        4     use data array as colors, if present        |
|        5-6   change background color(s)                  |
|        09+-  (on keypad) or +/- to cycle axes style      |
|        k     cycle available lighting styles             |
|        K     cycle available shading styles              |
|        A     toggle anti-aliasing                        |
|        D     toggle depth-peeling (for transparencies)   |
|        o/O   add/remove light to scene and rotate it     |
|        n     show surface mesh normals                   |
|        a     toggle interaction to Actor Mode            |
|        j     toggle interaction to Joystick Mode         |
|        u     toggle perspective/parallel projection      |
|        r     reset camera position                       |
|        C     print current camera settings               |
|        S     save a screenshot                           |
|        E     export rendering window to numpy file       |
|        q     return control to python script             |
|        Esc   abort execution and exit python kernel      |
|----------------------------------------------------------|
| Mouse: Left-click    rotate scene / pick actors          |
|        Middle-click  pan scene                           |
|        Right-click   zoom scene in or out                |
|        Cntrl-click   rotate scene                        |
 ==========================================================
```

### file format conversion
You can convert on the fly a file (or multiple files) to a different format with
```bash
vedo --convert bunny.obj --to ply
```

### some useful bash aliases
```bash
alias vr='vedo --run '        # to search and run examples by name
alias vs='vedo -i --search '  # to search for a string in examples
alias ve='vedo --eog '        # to view single and multiple images
alias vv='vedo -bg blackboard -bg2 gray3 -z 1.05 -k glossy -c blue9'
```

## Example Galleries
Check out the example galleries organized by subject [**here**](https://vedo.embl.es/#gallery).


## Running on a headless server
- Install `libgl1-mesa` and `xvfb` on your server:
```bash
sudo apt install libgl1-mesa-glx libgl1-mesa-dev xvfb
pip install vedo
```

- Execute on startup:
```bash
set -x
export DISPLAY=:99.0
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x
exec "$@"
```

- You can save the above code above as `/etc/rc.local` and use `chmod +x` to make it executable.
    It may throw an error during startup. Then test it with, e.g.:
```python
import vedo
plt = vedo.Plotter(offscreen=True, size=(500,500))
plt.show(vedo.Cube()).screenshot('mycube.png').close()
```

## Running in a Docker container
You need to set everything up for offscreen rendering: there are two main ingredients

- `vedo` should be set to render in offscreen mode
- guest OS in the docker container needs the relevant libraries installed
    (in this example we need the Mesa openGL and GLX extensions, and Xvfb to act as a virtual screen.
    It's maybe also possible to use OSMesa offscreen driver directly, but that requires a custom
    build of VTK).

- Create a `Dockerfile`:
```bash
FROM python:3.8-slim-bullseye

RUN apt-get update -y \
  && apt-get install libgl1-mesa-dev libgl1-mesa-glx xvfb -y --no-install-recommends \
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*
RUN pip install vedo && rm -rf $(pip cache dir)
RUN mkdir -p /app/data

WORKDIR /app/
COPY test.py set_xvfb.sh /app/
ENTRYPOINT ["/app/set_xvfb.sh"]
```

- `set_xvfb.sh`:
```bash
#!/bin/bash
set -x
export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
#sleep 3
set +x
exec "$@"
```

- `test.py`:
```python
from vedo import Sphere, Plotter, settings
settings.screenshotTransparentBackground = True
sph = Sphere(pos=[-5, 0, 0], c="r")
plt = Plotter(interactive=False, offscreen=True)
plt.show(sph)
plt.screenshot("./data/out.png", scale=2).close()
```

Then you can

1. `$ docker build -t vedo-test-local .`
2. `$ docker run --rm -v /some/path/output:/app/data vedo-test-local python test.py` (directory `/some/path/output` needs to exist)
3. There should be an `out.png` file in the output directory.


## Getting help
Check out the [**Github repository**](https://github.com/marcomusy/vedo)
for more information, where you can ask questions and report issues.
You are also welcome to post specific questions on the [**image.sc**](https://forum.image.sc/) forum.
"""
##### To generate documentation #######################################################
# cd Projects/vedo
# pdoc --html . --force -c lunr_search="{'fuzziness': 0, 'index_docstrings': True}"
######################################################################### pdoc excludes
__pdoc__ = {}
__pdoc__['embedWindow'] = False
__pdoc__['backends'] = False
__pdoc__['cli'] = False
__pdoc__['cmaps'] = False
__pdoc__['version'] = False
__pdoc__['pointcloud.Points.pointColors'] = False
__pdoc__['pointcloud.Points.cellColors'] = False
__pdoc__['pointcloud.Points.thinPlateSpline'] = False
__pdoc__['pointcloud.Points.warpByVectors'] = False
__pdoc__['pointcloud.Points.distanceToMesh'] = False
__pdoc__['dolfin.show'] = False
__pdoc__['pyplot.show'] = False


#######################################################################################

__author__     = "Marco Musy"
__license__    = "MIT"
__maintainer__ = "M. Musy"
__email__      = "marco.musy@embl.es"
__status__     = "dev"
__website__    = "https://github.com/marcomusy/vedo"

######################################################################## imports
import os
import sys
import vtk
import warnings
import logging
from deprecated import deprecated
import numpy as np
from numpy import sin, cos, sqrt, exp, log, dot, cross  # just because useful

#################################################
from vedo.version import _version as __version__
from vedo.utils import *
import vedo.settings as settings
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


##################################################################################
########################################################################## GLOBALS
vtk_version = [
    int(vtk.vtkVersion().GetVTKMajorVersion()),
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

# silence annoying messages
warnings.simplefilter(action="ignore", category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


################################################################################
installdir = os.path.dirname(__file__)
dataurl    = "https://vedo.embl.es/examples/data/"

plotter_instance = None
notebook_plotter = None
notebookBackend  = None

## fonts
fonts_path = os.path.join(installdir, "fonts/")
fonts = [_f.split(".")[0] for _f in os.listdir(fonts_path) if '.npz' not in _f]
fonts = list(sorted(fonts))

################################################################## deprecated
@deprecated(reason="\x1b[7m\x1b[1m\x1b[31;1mPlease use Plotter(backend='...')\x1b[0m")
def embedWindow(backend='ipyvtk', verbose=True):
    """DEPRECATED: Please use Plotter(backend='...').

    Function to control whether the rendering window is inside
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
