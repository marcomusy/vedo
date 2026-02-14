
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vedo/badges/version.svg)](https://anaconda.org/conda-forge/vedo)
[![Ubuntu 24.10 package](https://repology.org/badge/version-for-repo/ubuntu_24_10/vedo.svg)](https://repology.org/project/vedo/versions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4587871.svg)](https://doi.org/10.5281/zenodo.4587871)

![](https://vedo.embl.es/images/feats/teapot_banner.png)


A Python module for scientific analysis of 3D objects and
point clouds based on [VTK](https://www.vtk.org/) and [NumPy](http://www.numpy.org/).

Check out the [**GitHub repository**](https://github.com/marcomusy/vedo)
and the [**vedo main page**](https://vedo.embl.es).


## Install and Test
```bash
pip install vedo

# Or, install the latest development version with:
pip install -U git+https://github.com/marcomusy/vedo.git
```
Then run:
```python
import vedo
vedo.Cone().show(axes=1).close()
```
![](https://vedo.embl.es/images/feats/cone.png)


## Command Line Interface
The library includes a **C**ommand **L**ine **I**nterface.
For example, type in your terminal:

```bash
vedo --help
vedo https://vedo.embl.es/examples/data/panther.stl.gz
```
![](https://vedo.embl.es/images/feats/vedo_cli_panther.png)

Pressing `h` will then show a number of options to interact with your 3D scene:
```             
   i     print info about the last clicked object     
   I     print color of the pixel under the mouse     
   Y     show the pipeline for this object as a graph 
   <- -> use arrows to reduce/increase opacity        
   x     toggle mesh visibility                       
   w     toggle wireframe/surface style               
   l     toggle surface edges visibility              
   p/P   hide surface faces and show only points      
   1-3   cycle surface color (2=light, 3=dark)        
   4     cycle color map (press shift-4 to go back)   
   5-6   cycle point-cell arrays (shift to go back)   
   7-8   cycle background and gradient color          
   09+-  cycle axes styles (on keypad, or press +/-)  
   k     cycle available lighting styles              
   K     toggle shading as flat or phong              
   A     toggle anti-aliasing                         
   D     toggle depth-peeling (for transparencies)    
   U     toggle perspective/parallel projection       
   o/O   toggle extra light to scene and rotate it    
   a     toggle interaction to Actor Mode             
   n     toggle surface normals                       
   r     reset camera position                        
   R     reset camera to the closest orthogonal view  
   .     fly camera to the last clicked point         
   C     print the current camera parameters state    
   X     invoke a cutter widget tool                  
   S     save a screenshot of the current scene       
   E/F   export 3D scene to numpy file or X3D         
   q     return control to python script              
   Esc   abort execution and exit python kernel       
```

### Some useful bash aliases
```bash
alias vr='vedo --run '      # to search and run examples by name
alias vs='vedo --search '   # to search for a string in examples
alias ve='vedo --eog '      # to view single and multiple images
```

## Tutorials
You are welcome to ask specific questions on the
[**image.sc**](https://forum.image.sc) forum,
open a [**GitHub issue**](https://github.com/marcomusy/vedo/issues)
or search the [**examples gallery**](https://vedo.embl.es/#gallery)
for some relevant example.

You can also find online tutorials at:

- [Vedo tutorial for the EMBL Python User Group](https://github.com/marcomusy/vedo-epug-tutorial) with [slides](https://github.com/marcomusy/vedo-epug-tutorial/blob/main/vedo-epug-seminar.pdf) by M. Musy (EMBL).

- [Summer School on Computational Modelling of Multicellular Systems](https://github.com/LauAvinyo/vedo-embo-course) with [slides](https://github.com/LauAvinyo/vedo-embo-course/blob/main/vedo-embo-presentation.pdf) by Laura Avinyo (EMBL).

- YouTube video tutorials by [M. El Amine](https://github.com/amine0110/pycad):
   - [Visualizing Multiple 3D Objects in Medical Imaging](https://www.youtube.com/watch?v=LVoj3poN2WI)
   - [Capture 3D Mesh Screenshots in Medical Imaging](https://www.youtube.com/watch?v=8Qn14WMUamA)
   - [Slice 'n Dice: Precision 3D Mesh Cutting](https://www.youtube.com/watch?v=dmXC078ZOR4&t=195s)
   - [3D Visualization of STL Files](https://www.youtube.com/watch?v=llq9-oJXepQ)


- [Creating an interactive 3D geological model](https://www.youtube.com/watch?v=raiIft8VeRU&t=1s) by A. Pollack (SCRF). See a more recent example
[here](https://github.com/marcomusy/vedo/blob/master/examples/advanced/geological_model.py).

- ["vedo", a python module for scientific analysis and visualization of 3D data](https://www.youtube.com/watch?v=MhIoetdxwc0&t=39s), I2K Conference, by M. Musy (EMBL).


## Export a 3D scene to file
You can export to a vedo file (a standard `numpy` file) by pressing `E`
in your 3D scene. You can then interact with it normally, for example using the key bindings shown above.

Another option is to export to a template HTML web page by pressing `F` with the `x3d` backend.
You can also export programmatically in `k3d` format from a Jupyter notebook.


## File format conversion
You can convert on the fly a file (or multiple files) to a different format with
```bash
vedo --convert bunny.obj --to ply
```


## Running in a Jupyter Notebook
To use vedo in Jupyter notebooks, set `vedo.settings.default_backend = "..."`.
The supported backends are:

- `2d`, the default. A static image is generated.
- `vtk`, in this case a normal graphics rendering window will pop up.
- [k3d](https://github.com/K3D-tools/K3D-jupyter), use with `pip install k3d`.
- [ipyvtklink](https://github.com/Kitware/ipyvtklink) (allows interaction with the scene).
- [trame](https://www.kitware.com/trame-visual-analytics-everywhere/)

Check for more examples in the
[repository](https://github.com/marcomusy/vedo/tree/master/examples/notebooks).

### Running on Google Colab
Start your notebook with:
```python
import vedo
vedo.settings.init_colab()
```

Then test it with:
```python
import vedo
print("vedo", vedo.__version__)
sphere = vedo.Sphere().linewidth(1)
plt = vedo.Plotter()
plt += sphere
plt.show(axes=1, viewup='z', zoom=1.5)
```


## Running on a Server
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

- You can save the above startup script as `/etc/rc.local` and use `chmod +x` to make it executable.
    If it reports an error during startup, test it with, e.g.:
```python
import vedo
plt = vedo.Plotter(offscreen=True, size=(500,500))
plt.show(vedo.Cube()).screenshot('mycube.png').close()
```

## Running in a Docker container
You need to set everything up for offscreen rendering. There are two main requirements:

- `vedo` should be set to render in offscreen mode.
- The guest OS in the Docker container needs the relevant libraries installed
    (in this example we need the Mesa openGL and GLX extensions, and Xvfb to act as a virtual screen.
    It may also be possible to use the OSMesa offscreen driver directly, but that requires a custom
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
sph = Sphere(pos=[-5, 0, 0], c="r")
plt = Plotter(interactive=False, offscreen=True)
plt.show(sph)
plt.screenshot("./data/out.png", scale=2).close()
```

Then you can:

1. `$ docker build -t vedo-test-local .`
2. `$ docker run --rm -v /some/path/output:/app/data vedo-test-local python test.py` (directory `/some/path/output` needs to exist)
3. There should be an `out.png` file in the output directory.


## Generate a single executable file
You can use [pyinstaller](https://pyinstaller.readthedocs.io/en/stable/)
to generate a single portable executable file for different platforms.

Write a file `myscript.spec` as:
```python
# -*- mode: python ; coding: utf-8 -*-
#
import os

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from vedo import installdir as vedo_installdir
vedo_fontsdir = os.path.join(vedo_installdir, 'fonts')
print('vedo installation is in', vedo_installdir)
print('fonts are in', vedo_fontsdir)

block_cipher = None

added_files = [
    (os.path.join('tuning','*'), 'tuning'),
    (os.path.join(vedo_fontsdir,'*'), os.path.join('vedo','fonts')),
]

a = Analysis(['myscript.py'],
             pathex=[],
             binaries=[],
             hiddenimports=[
                 'vtkmodules',
                 'vtkmodules.all',
                 'vtkmodules.util',
                 'vtkmodules.util.numpy_support',
                 'vtkmodules.qt.QVTKRenderWindowInteractor',
             ],
             datas = added_files,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='my_program_name',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)

```
Then run it with:
```bash
pyinstaller myscript.spec
```
See also an example [here](https://github.com/marcomusy/welsh_embryo_stager/blob/main/stager.spec).

If you get an [error message](https://github.com/marcomusy/vedo/discussions/820) related to a font that is not shipped with vedo, copy the `.npz` and `.ttf` files to `vedo/fonts` (where the other fonts are) and reinstall vedo.
Then add in your script `settings.font_parameters["FONTNAME"]["islocal"] = True`.
