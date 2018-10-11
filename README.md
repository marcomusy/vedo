# vtkplotter
A python class to easily draw, analyse and animate 3D objects.

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


## Usage examples
Download and Run the tutorial:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  # on mac OSX try 'pythonw' instead
```

### Simple command-line usage:
```bash
vtkplotter data/*.vtk  # on Windows try 'python vtkplotter'
# other valid formats: [vtu,vts,vtp, ply,obj,stl,xml,neutral,gmsh,pcd,xyz,txt,byu, tif,slc, png,jpg]
```
to visualize multiple files or files time-sequences try `-n` or `-s` options.<br />
Voxel-data (*slc*, *tiff* stack) files can also be visualized with option `-g`, e.g.:
```bash
vtkplotter -g -c blue examples/data/embryo.slc
```

```
usage: vtkplotter files [-h] [-a] [-w] [-p] [-l] [-c] [-k] [-n] [-x] [-f] [-z] [-i] [-b] [-q] [-s]

positional arguments:
  files                 Input filename(s)

optional arguments:
  -h, --help            show this help message and exit
  -a, --alpha           alpha value [0-1]
  -w, --wireframe       use wireframe representation
  -p, --point-size      specify point size
  -l, --legend-off      do not show legends
  -c, --color           mesh color [integer or color name]
  -k, --show-scalars    use scalars as color
  -x, --axes-type       specify axes type [0-3]
  -f, --full-screen     full screen mode
  -z, --zoom            zooming factor
  -i, --no-camera-share do not share camera in renderers
  -b, --background      background color [integer or color name]
  -q, --quiet           quiet mode, less verbose
  -n, --sequence-mode   show each file in a separate renderer
  -s, --scrolling-mode  Scrolling Mode: use arrows to scroll files
  -g, --ray-cast-mode   GPU Ray-casting Mode for SLC/TIFF files
  -gz, --z-spacing      Volume z-spacing factor [1]
  -gy, --y-spacing      Volume y-spacing factor [1]
```
<br />

The command `vtkconvert -to ply file.vtk` can be used to convert mesh formats easily:
```
usage: vtkconvert [-h] [-to] [files [files ...]]

Allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]
```
<br />


From within your python script, load a simple OBJ file and display it:
```python
from vtkplotter import Plotter

vp = Plotter()              # declare an instance of the class
vp.show('data/shuttle.obj') # press *Esc* to close and exit or *q* to continue
```
![shuttle](https://user-images.githubusercontent.com/32848391/35975974-e1235396-0cde-11e8-9880-69335cc7fd43.png)
<br />


Draw a bunch of basic geometric objects on separate parts of the rendering window:
```python
vp = Plotter(N=6, sharecam=False)
vp.show(vp.arrow([0,0,0], [1,1,1]),   at=0, legend='arrow()' )
vp.show(vp.line([0,0,0], [1,1,1]),    at=1, legend='line()' )
vp.show(vp.point([1,2,3]),            at=2, legend='point()' )
vp.show(vp.text('Hello', bc=(1,0,0)), at=3 )
vp.show(vp.sphere(),                  at=4 )
vp.show(vp.cube(),                    at=5, legend='cube()')
vp.show(interactive=1)
```
![ex8](https://user-images.githubusercontent.com/32848391/32666975-91690102-c639-11e7-8f7b-ad07bd6019da.png)
<br />


Load meshes of various formats with different options:
```python
vp = Plotter(shape=(3,3), sharecam=False)
vp.show(at=0, c=0, actors='data/beethoven.ply', ruler=1, axes=0)
vp.show(at=1, c=1, actors='data/cow.g', wire=1)
vp.show(at=2, c=2, actors='data/limb.pcd') # point cloud format (pcl)
vp.show(at=3, c=3, actors='data/shapes/spider.ply')
vp.show(at=4, c=4, actors='data/shuttle.obj')
vp.show(at=5, c=5, actors='data/shapes/magnolia.vtk')
vp.show(at=6, c=6, actors='data/shapes/man.vtk', axes=1)
vp.show(at=7, c=7, actors='data/teapot.xyz', axes=2)
vp.show(at=8, actors='data/pulley.vtu', axes=3)
vp.show(interactive=1)
```
![objects](https://user-images.githubusercontent.com/32848391/43654734-8d126a96-974c-11e8-80d6-73cf224c0511.png)
<br />


Plot *f(x,y) = sin(3*x)*log(x-y)/3.
Red dots indicate where function *f* is not a real number
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fxy.py)):
```python
vp = Plotter()  
vp.fxy('sin(3*x)*log(x-y)/3')
vp.show()
```
![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png)
<br />



Apply Moving Least Squares algorithm to a large point cloud to obtain a smooth surface 
from a set of scattered points in space ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)):
```bash
python examples/advanced/moving_least_squares2D.py
```
![bunnymls](https://user-images.githubusercontent.com/32848391/43954472-ef161148-9c9c-11e8-914d-1ba57718da74.png)
<br />



Motion of particles of gas in a toroidal tank. The spheres collide elastically with themselves and
with the walls of the tank ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gas.py)):
```bash
python examples/advanced/gas.py
```
![gas](https://user-images.githubusercontent.com/32848391/39139206-90d644ca-4721-11e8-95b9-8aceeb3ac742.gif)
<br />



Simulation of an elastic multiple pendulum with friction ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/multiple_pendulum.py)):
```bash
python examples/advanced/multiple_pendulum.py
```
![pendulum](https://user-images.githubusercontent.com/32848391/39259507-dc26b18a-48b6-11e8-94fd-3fcb01661b55.gif)
<br />


Direct integration of the wave equation comparing the simple Euler method in green
with the more sofisticated Runge-Kutta 4th order method in red 
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/wave_equation.py)):
```bash
python examples/advanced/wave_equation.py
```
![wave](https://user-images.githubusercontent.com/32848391/39360796-ea5f9ef0-4a1f-11e8-85cb-f3e21072c7d5.gif)
<br />



Simulation of a gyroscope hanging from a spring
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)):
```bash
python examples/advanced/gyroscope1.py
```
![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)
<br />


Simulation of [Rutherford scattering](https://en.wikipedia.org/wiki/Rutherford_scattering) 
of charged particles on a fixed target (by T. Vandermolen,
[script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/particle_simulator.py)):
```bash
python examples/advanced/particle_simulator.py 
```
![ruth](https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif)
<br />


Visualizing a Turing system of reaction-diffusion between two molecules
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)):
```bash
python examples/advanced/turing.py
```
![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)
<br />


More examples can be found in directories: **examples/basic** and **examples/advanced**.

If you need to do more complicated things (define widgets.. etc), you still have full access 
to all standard VTK objects (e.g. interactors and renderers 
through *vp.interactor, vp.renderer*... etc).<br />
In linux systems with *ffmpeg* you can use *vp.openVideo(), 
video.addFrame()* and *video.close()* to save a *movie.mp4* file.
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
Theese methods return the actor object so that they can be concatenated,
(e.g.: `actor.scale(3).pos([1,2,3]).color('blue').alpha(0.5)` etc..).
```python
actor.pos()                   # set/get position vector (setters, and getters if no argument is given)
actor.addpos(v)               # add v to current actor position
actor.x()                     # set/get x component of position (same for y and z)
actor.scale()                 # set/get scaling factor of actor
actor.normalize()             # sets actor at origin and scales its average size to 1
actor.color(value)            # sets/gets color
actor.alpha(value)            # sets/gets opacity
actor.N()                     # get number of vertex points defining the surface actor
actor.polydata()              # get the actor's mesh polydata including its current transformation
actor.coordinates()           # get a copy of vertex points coordinates (use copy=False to get references)
actor.normals()               # get the list of normals at the vertices of the surface
actor.xbounds()               # get (xmin, xmax) of actor bounding box (same for y and z)
actor.rotate(angle, axis, axis_point=[0,0,0], rad=False)  # rotate actor around axis
actor.clone(c=None, alpha=None, wire=False, bc=None, edges=False, legend=None, texture=None)
```

### Some useful shortcuts available in vtkplotter (*a la vpython*)
```python
def vector(x, y=None, z=0)    # return a numpy vector (2D or 3D)
def mag(v)                    # return the size of a vector or list of vectors
def mag2(v)                   # return the square of the size of a vector 
def norm(v)                   # return the versor of a vector or list of vectors
```

### Available color maps from matplotlib
```python
# Example: transform a scalar value between -10.2 and 123.4 into a RGB color, using the 'jet' map
from vtkplotter import colorMap
RGBcol = colorMap(value, name='jet', vmin=-10.2, vmax=123.4)
```
![colmaps](https://user-images.githubusercontent.com/32848391/42942959-c8b50eec-8b61-11e8-930a-00dcffdca601.png)

<br />
Tested on VTK versions 6.3, 7.1, 8.1<br />
