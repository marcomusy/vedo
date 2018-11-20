# vtkplotter

A python class to easily draw, analyse and animate 3D objects with [VTK](https://www.vtk.org/) 
and [numpy](http://www.numpy.org/).


## Download and Install:
After installing [VTK](https://www.vtk.org/) (e.g. with 
`conda install -c conda-forge vtk`
or `sudo apt install vtk7` 
or `pip install vtk`), simply type:
```bash
pip install --upgrade vtkplotter
```


## Documentation
Complete module documentation can be found [**here**](https://vtkplotter.embl.es).



## Basic usage examples
From within your python script, load a simple 3DS file and display it:
```python
from vtkplotter import Plotter

vp = Plotter()                      # declare an instance of the class
vp.show('data/shapes/flamingo.3ds') # press *Esc* to close and exit or *q* to continue
```
![flam](https://user-images.githubusercontent.com/32848391/47579206-9b88ab80-d94b-11e8-9f04-132698fd5ea3.png)
<br />


Load meshes of various formats with different options:
```python
vp = Plotter(shape=(3,3), sharecam=False)
vp.show('data/beethoven.ply',      at=0, c=0, ruler=1, axes=0)
vp.show('data/cow.g',              at=1, c=1, wire=1)
vp.show('data/limb.pcd',           at=2, c=2) # point cloud format (pcl)
vp.show('data/shapes/spider.ply',  at=3, c=3)
vp.show('data/shuttle.obj',        at=4, c=4)
vp.show('data/shapes/magnolia.vtk',at=5, c=5)
vp.show('data/shapes/man.vtk',     at=6, c=6, axes=1)
vp.show('data/teapot.xyz',         at=7, c=7, axes=2)
vp.show('data/pulley.vtu',         at=8, c=8, axes=3)
vp.show(interactive=1)
```
![objects](https://user-images.githubusercontent.com/32848391/43654734-8d126a96-974c-11e8-80d6-73cf224c0511.png)
<br />

Draw a bunch of basic geometric objects on separate parts of the rendering window:
```python
from vtkplotter.shapes import arrow, line, point, text, sphere, cube
vp = Plotter(N=6, sharecam=False)  # subdivide window in 6 independent parts
vp.show(arrow([0,0,0], [1,1,1]),   at=0, legend='an arrow')
vp.show(line( [0,0,0], [1,1,1]),   at=1, legend='a line')
vp.show(point([1,2,3], r=20),      at=2, legend='a point')
vp.show(text('Hello', bc=(1,0,0)), at=3)
vp.show(sphere(),                  at=4)
vp.show(cube(),                    at=5, legend='a cube')
vp.show(interactive=1)
```
![ex8](https://user-images.githubusercontent.com/32848391/32666975-91690102-c639-11e7-8f7b-ad07bd6019da.png)
<br />


If you need to do more complicated things (define widgets.. etc), you still have full access 
to all standard VTK native objects 
(e.g. interactors and renderers through *vp.interactor, vp.renderer, vtkActor*... etc).
<br />



### Command-line usage:
```bash
vtkplotter meshfile.vtk  # on Windows try 'python vtkplotter'
# other valid formats: [vtu,vts,vtp, ply,obj,stl,xml,neutral,gmsh,pcd,xyz,txt,byu, tif,slc, png,jpg]
```
to visualize multiple files or files time-sequences try `-n` or `-s` options.<br />
Voxel-data (*slc*, *tiff* stack) files can also be visualized with options `-g` and `--slicer`,
e.g.:
```bash
vtkplotter -g -c blue examples/data/embryo.slc  # (3D scan of a mouse embryo)
vtkplotter --slicer   examples/data/embryo.slc    
```
![e2](https://user-images.githubusercontent.com/32848391/48278506-00fd9180-e44e-11e8-94e6-6ee5f2a56ff7.jpg)

```
usage: vtkplotter files [-h] [-a] [-w] [-p] [-l] [-c] [-k] [-n] [-x] [-f] [-z] [-i] [-b] [-q] [-s]

positional arguments:
  files                 Input filename(s)

optional arguments:
  -h, --help            show this help message and exit
  -a , --alpha          alpha value [0-1]
  -w, --wireframe       use wireframe representation
  -p , --point-size     specify point size
  -l, --legend-off      do not show legends
  -c , --color          mesh color [integer or color name]
  -k, --show-scalars    use scalars as colors
  -x , --axes-type      specify axes type [0-3]
  -f, --full-screen     full screen mode
  -z , --zoom           zooming factor
  -i, --no-camera-share  do not share camera in renderers
  -b , --background     background color [integer or color name]
  -q, --quiet           quiet mode, less verbose
  -n, --sequence-mode   show each file in a separate renderer
  -s, --scrolling-mode  Scrolling Mode: use arrows to scroll files
  -g, --ray-cast-mode   GPU Ray-casting Mode for SLC/TIFF files
  -gz , --z-spacing     Volume z-spacing factor [1]
  -gy , --y-spacing     Volume y-spacing factor [1]
  --slicer              Slicer Mode for SLC/TIFF files
```
<br />


## Examples Gallery
A get-started tutorial script is available for download:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  # on mac OSX try 'pythonw' instead
```
Many more examples can be found in directories: **examples/basic**, **examples/advanced**
and **examples/volumetric**.
<br />


- Apply a Moving Least Squares algorithm to a large point cloud to obtain a smooth surface 
from a set of scattered points in space 
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)):<br />
`python examples/advanced/moving_least_squares2D.py`<br />
![bunnymls](https://user-images.githubusercontent.com/32848391/43954472-ef161148-9c9c-11e8-914d-1ba57718da74.png)
<br />


- Motion of particles of gas in a toroidal tank. The spheres collide elastically with themselves and
with the walls of the tank 
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gas.py)):<br />
`python examples/advanced/gas.py`<br />
![gas](https://user-images.githubusercontent.com/32848391/39139206-90d644ca-4721-11e8-95b9-8aceeb3ac742.gif)
<br />


- Simulation of a gyroscope hanging from a spring
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)):<br />
`python examples/advanced/gyroscope1.py`<br />
![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)
<br />


- Simulation of [Rutherford scattering](https://en.wikipedia.org/wiki/Rutherford_scattering) 
of charged particles on a fixed target (by T. Vandermolen,
[script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/particle_simulator.py)):<br />
`python examples/advanced/particle_simulator.py`<br />
![ruth](https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif)
<br />


- Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. 
The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/tunnelling2.py)):<br />
`python examples/advanced/tunnelling2.py`<br />
![qsine2](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)
<br />


- Visualizing a Turing system of reaction-diffusion between two molecules
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)):<br />
`python examples/advanced/turing.py`<br />
![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)
<br />



### Some useful *Plotter* attributes
```python
vp = vtkplotter.Plotter() #e.g.
vp.actors       # holds the current list of vtkActors to be shown
vp.renderer     # holds the current renderer
vp.renderers    # holds the list of renderers
vp.interactor   # holds the vtkWindowInteractor object
vp.interactive  # (True) allows to interact with renderer after show()
vp.camera       # holds the current vtkCamera
vp.sharecam     # (True) share the same camera in multiple renderers
```


### Some useful additional methods of vtkActor object
These methods return the Actor object so that they can be concatenated,
(E.g.: `actor.scale(3).pos([1,2,3]).color('blue').alpha(0.5)` etc..).
```python
actor.pos()                   # set/get position vector (setters, and getters if no argument is given)
actor.addpos(v)               # add v to current actor position
actor.x()                     # set/get x component of position (same for y and z)
actor.scale()                 # set/get scaling factor of actor
actor.normalize()             # sets actor at origin and scales its average size to 1
actor.color(name)             # sets/gets color
actor.alpha(value)            # sets/gets opacity
actor.N()                     # get number of vertex points defining the actor's mesh
actor.polydata(True)          # get the actor's mesh polydata in its current transformation (True_
actor.coordinates()           # get a copy of vertex points coordinates (set copy=False to get references)
actor.normals()               # get the list of normals at the vertices of the surface
actor.xbounds()               # get (xmin, xmax) of actor bounding box (same for y and z)
actor.rotate(angle, axis, axis_point=[0,0,0], rad=False)  # rotate actor around axis
actor.clone(c=None, alpha=None, wire=False, bc=None, edges=False, legend=None, texture=None)
...
```


### Mesh format conversion
The command `vtkconvert` can be used to convert a file format easily:
```
usage: vtkconvert [-h] [-to] [files [files ...]]

Allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]

Example: vtkconvert -to ply myfile.vtk
```


### Available color maps from *matplotlib* and *vtkNamedColors*
```python
# Example: transform a scalar value between -10.2 and 123.4 into a (R,G,B) color using the 'jet' map:
from vtkplotter import colorMap
r, g, b = colorMap(value, name='jet', vmin=-10.2, vmax=123.4)
```
![colmaps](https://user-images.githubusercontent.com/32848391/42942959-c8b50eec-8b61-11e8-930a-00dcffdca601.png)

A list of available vtk color names is given [here](https://vtkplotter.embl.es/vtkcolors.html).
<br />
