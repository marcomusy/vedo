
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vedo/badges/version.svg)](https://anaconda.org/conda-forge/vedo)
[![Ubuntu 23.04 package](https://repology.org/badge/version-for-repo/ubuntu_23_04/vedo.svg)](https://repology.org/project/vedo/versions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5842090.svg)](https://doi.org/10.5281/zenodo.5842090)

![](https://user-images.githubusercontent.com/32848391/46815773-dc919500-cd7b-11e8-8e80-8b83f760a303.png)


A python module for scientific analysis of 3D objects and
point clouds based on [VTK](https://www.vtk.org/) and [numpy](http://www.numpy.org/).

Check out the [GitHub repository here](https://github.com/marcomusy/vedo).

## Install and Test
```bash
pip install vedo
# Or, install the latest development version:
pip install -U git+https://github.com/marcomusy/vedo.git
```
Then
```python
import vedo
vedo.Cone().show(axes=1).close()
```
![](https://vedo.embl.es/images/feats/cone.png)


## Command Line Interface
The library includes a **C**ommand **L**ine **I**nterface.
Type for example in your terminal:

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
    C     print current camera settings                
    X     invoke a cutter widget tool                  
    S     save a screenshot of the current scene       
    E/F   export 3D scene to numpy file or X3D         
    q     return control to python script              
    Esc   abort execution and exit python kernel       
```


## Export your 3D scene to file
You can export it to a vedo file, which is actually a normal `numpy` file by pressing `E`
in your 3D scene, the you can interact with it normally using for example the key bindings shown above.

Another way is to export to a template html web page by pressing `F` using `x3d` backend.
You can also export it programmatically in `k3d` from a jupyter notebook.


## File format conversion
You can convert on the fly a file (or multiple files) to a different format with
```bash
vedo --convert bunny.obj --to ply
```

### Some useful bash aliases
```bash
alias vr='vedo --run '        # to search and run examples by name
alias vs='vedo -i --search '  # to search for a string in examples
alias ve='vedo --eog '        # to view single and multiple images
```

## Running in a Jupyter Notebook
To use in jupyter notebooks use the syntax `vedo.settings.default_backend= '...' ` the supported backend for visualization are:

- `2d`, the default a static image is generated.
- `vtk`, in this case a normal graphics rendering window will pop up.
- [k3d](https://github.com/K3D-tools/K3D-jupyter) use with `pip install k3d`
- [ipyvtklink](https://github.com/Kitware/ipyvtklink) (allows interaction with the scene).
- [trame](https://www.kitware.com/trame-visual-analytics-everywhere/)

Check for more examples in 
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
sph = Sphere(pos=[-5, 0, 0], c="r")
plt = Plotter(interactive=False, offscreen=True)
plt.show(sph)
plt.screenshot("./data/out.png", scale=2).close()
```

Then you can

1. `$ docker build -t vedo-test-local .`
2. `$ docker run --rm -v /some/path/output:/app/data vedo-test-local python test.py` (directory `/some/path/output` needs to exist)
3. There should be an `out.png` file in the output directory.


## Generate a single executable file
You can use [pyinstaller](https://pyinstaller.readthedocs.io/en/stable/)
to generate a single, portable, executable file for different platforms.

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
          name='myprogramname',
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
then run it with
```bash
pyinstaller myscript.spec
```
See also an example [here](https://github.com/marcomusy/welsh_embryo_stager/blob/main/stager.spec).

If you get an [error message](https://github.com/marcomusy/vedo/discussions/820) related to a font which is not shipped with the vedo library you will need to copy the `.npz` and `.ttf` files to `vedo/fonts` (where all the other fonts are) and reinstall vedo. 
Then add in your script `settings.font_parameters["FONTNAME"]["islocal"] = True`.


<!-- .. include:: ../docs/tutorials.md -->

## Getting help
Check out the [**Github repository**](https://github.com/marcomusy/vedo)
for more information, where you can ask questions and report issues.
You are also welcome to post specific questions on the [**image.sc**](https://forum.image.sc/) forum,
or simply browse the [**examples gallery**](https://vedo.embl.es/#gallery).

You can also find online tutorials at:

- [Summer School on Computational Modelling of Multicellular Systems](https://github.com/LauAvinyo/vedo-embo-course) with [slides](https://github.com/LauAvinyo/vedo-embo-course/blob/main/vedo-embo-presentation.pdf) by Laura Avinyo (EMBL).

- Youtube video tutorials: 
[Visualizing Multiple 3D Objects in Medical Imaging](https://www.youtube.com/watch?v=LVoj3poN2WI),
[Capture 3D Mesh Screenshots in Medical Imaging](https://www.youtube.com/watch?v=8Qn14WMUamA),
[Slice 'n Dice: Precision 3D Mesh Cutting](https://www.youtube.com/watch?v=dmXC078ZOR4&t=195s),
[3D Visualization of STL Files](https://www.youtube.com/watch?v=llq9-oJXepQ)
by [M. El Amine](https://github.com/amine0110/pycad).

- [Creating an interactive 3D geological model](https://www.youtube.com/watch?v=raiIft8VeRU&t=1s) by A. Pollack.

- ["vedo", a python module for scientific analysis and visualization of 3D data](https://www.youtube.com/watch?v=MhIoetdxwc0&t=39s), I2K Conference, by M. Musy (EMBL).


