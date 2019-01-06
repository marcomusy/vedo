# vtkplotter
[![Downloads](https://pepy.tech/badge/vtkplotter)](https://pepy.tech/project/vtkplotter)
[![Downloads](https://pepy.tech/badge/vtkplotter/week)](https://pepy.tech/project/vtkplotter)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![pythvers](https://img.shields.io/badge/python-2.7%7C3.6-brightgreen.svg)](https://pypi.org/project/vtkplotter)

A python module to easily draw, analyse and animate 3D objects with [VTK](https://www.vtk.org/) 
and [numpy](http://www.numpy.org/).


## Download and Install:
Simply type:
```bash
pip install vtkplotter
```

## Documentation
Complete module documentation can be found [**here**](https://vtkplotter.embl.es).
<br />
<br />


## Get-started example
- In your python script, load a simple `3DS` file and display it:
```python
from vtkplotter import Plotter

vp = Plotter()                      # declare an instance of the class
vp.show('data/shapes/flamingo.3ds') # press *Esc* to close and exit or *q* to continue
```
![flam](https://user-images.githubusercontent.com/32848391/50738813-58af4380-11d8-11e9-84ce-53579c1dba65.png)
<br />
<br />


- Load meshes of various formats with different options on separate parts of the rendering window:
```python
vp = Plotter(shape=(2,3), sharecam=False) # subdivide window in 6 independent parts
vp.show('data/beethoven.ply',      at=0, c=0) # c = color name or number
vp.show('data/cow.g',              at=1, c=1, wire=1)
vp.show('data/limb.pcd',           at=2, c=2) # point cloud format (.pcl)
vp.show('data/shapes/spider.ply',  at=3, c=3)
vp.show('data/shuttle.obj',        at=4, c=4)
vp.show('data/shapes/magnolia.vtk',at=5, c=5)
vp.show(interactive=1)
```
![shapes](https://user-images.githubusercontent.com/32848391/50738812-58af4380-11d8-11e9-96d6-cc3780c2bac2.jpg)
<br />
<br />


Draw a bunch of basic geometric objects:
```python
from vtkplotter.shapes import arrow, line, point, text, sphere, cube
vp = Plotter(N=6, sharecam=False)  # automatically subdivide window in 6 independent parts
vp.show(arrow([0,0,0], [1,1,1]),   at=0)
vp.show(line( [0,0,0], [1,1,1]),   at=1)
vp.show(point([1,2,3], r=20),      at=2)
vp.show(text('Hello', bc=(1,0,0)), at=3)
vp.show(sphere(),                  at=4)
vp.show(cube(),                    at=5, interactive=True)
```
![ex8](https://user-images.githubusercontent.com/32848391/50738811-58af4380-11d8-11e9-9bfb-378c27c9d26f.png)
<br />

## Command-line usage
```bash
vtkplotter meshfile.vtk 
# other valid formats: [vtu,vts,vtp,ply,obj,stl,xml,neutral,gmsh,pcd,xyz,txt,byu,tif,slc,vti,png,jpg]
```
to visualize multiple files or files time-sequences try `-n` or `-s` options.<br />. Try `-h` for help.
Voxel-data (vti, slc, tiff) files can also be visualized with options `-g` and `--slicer`,
e.g.:
```bash
vtkplotter -g -c blue examples/data/embryo.slc  # (3D scan of a mouse embryo)
vtkplotter --slicer   examples/data/embryo.slc    
```
![e2](https://user-images.githubusercontent.com/32848391/50738810-58af4380-11d8-11e9-8fc7-6c6959207224.jpg)
<br />
<br />
<br />


## Examples Gallery
A get-started tutorial script is available for download:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  # on mac OSX try 'pythonw' instead
```
**Many more examples can be found in directories:** <br>
[**examples/basic**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic), 
[**examples/advanced**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced),
[**examples/volumetric**](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric)
and [**examples/others**](https://github.com/marcomusy/vtkplotter/blob/master/examples/others).

|    |    |
|:-------------:|:-----|
| ![rabbit](https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg)  | Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a large cloud of scattered points in space ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)) <br />  `python advanced/moving_least_squares2D.py` |
|    |    |
| ![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif) | Simulation of a gyroscope hanging from a spring ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)) <br /> `python advanced/gyroscope1.py`|
|    |    |
|  ![ruth](https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif)  | Simulation of [Rutherford scattering](https://en.wikipedia.org/wiki/Rutherford_scattering) of charged particles on a fixed target ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/particle_simulator.py))<br /> `python advanced/particle_simulator.py`   |
|    |    |
| ![qsine2](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)   | Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier. ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/tunnelling2.py)) <br /> `python advanced/tunnelling2.py`   |
|    |    |
| ![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)   |  Visualizing a Turing system of reaction-diffusion between two molecules ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)) <br /> `python advanced/turing.py`  |
<br />

### Some useful *Plotter* attributes
Remember that you always have full access to all standard VTK native objects 
(e.g. vtkRenderWindowInteractor, vtkRenderer and vtkActor through *vp.interactor, vp.renderer, vp.actors*... etc).
```python
vp = vtkplotter.Plotter() #e.g.
vp.actors       # holds the current list of vtkActors to be shown
vp.renderer     # holds the current vtkRenderer
vp.renderers    # holds the list of renderers
vp.interactor   # holds the vtkWindowInteractor object
vp.interactive  # (True) allows to interact with renderer after show()
vp.camera       # holds the current vtkCamera
vp.sharecam     # (True) share the same camera in multiple renderers
```

### Some useful additional methods to manage 3D objects
These methods return the Actor(vtkActor) object so that they can be concatenated,
check out [Actor methods here](https://vtkplotter.embl.es/actors.m.html). <br />
(E.g.: `actor.scale(3).pos([1,2,3]).color('blue').alpha(0.5)` etc..).
```python
actor.pos()               # set/get position vector (setters, and getters if no argument is given)
actor.x()                 # set/get x component of position (same for y and z)
actor.scale()             # set/get scaling factor of actor
actor.normalize()         # sets actor at origin and scales its average size to 1
actor.rotate(angle, axis) # rotate actor around axis
actor.color(name)         # sets/gets color
actor.alpha(value)        # sets/gets opacity
actor.N()                 # get number of vertex points defining the actor's mesh
actor.polydata()          # get the actor's mesh polydata in its current transformation
actor.coordinates()       # get a copy of vertex points coordinates (copy=False to get references)
actor.normals()           # get the list of normals at the vertices of the surface
actor.clone()             # get a copy of actor
...
```


### Mesh format conversion
The command `vtkconvert` can be used to convert multiple files from a format to a different one:
```
Usage: vtkconvert [-h] [-to] [files [files ...]]
allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]

Example: > vtkconvert myfile.vtk -to ply
```


### Available color maps from *matplotlib* and *vtkNamedColors*
```python
# Example: transform a scalar value between -10.2 and 123 into a (R,G,B) color using the 'jet' map:
from vtkplotter import colorMap
r, g, b = colorMap(value, name='jet', vmin=-10.2, vmax=123)
```
![colormaps](https://user-images.githubusercontent.com/32848391/50738804-577e1680-11d8-11e9-929e-fca17a8ac6f3.jpg)

A list of available vtk color names is given [here](https://vtkplotter.embl.es/vtkcolors.html).
<br />
