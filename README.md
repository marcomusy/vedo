# vtkplotter
A python helper class to easily draw, analyse and animate 3D objects.

## Download and Install:
After installing [VTK](https://www.vtk.org/) (e.g. with 
`conda install -c conda-forge vtk`
or `sudo apt install vtk7` 
or `pip install vtk`), simply type:
```bash
pip install --upgrade vtkplotter
```

## Usage examples
Download and Run the tutorials:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  # on mac OSX try 'pythonw' instead
```

Simple command-line usage:
```bash
vtkplotter data/*.vtk  # on Windows try python vtkplotter
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
```
<br />

The command `vtkconvert -to ply file.vtk` can be used to convert file formats easily:
```
usage: vtkconvert [-h] [-to] [files [files ...]]

Allowed targets: ['vtk', 'vtp', 'vtu', 'vts', 'ply', 'stl', 'byu', 'xml']
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

Load 3 actors assigning each a different color, use their file names as legend entries.<br />
(the 3D shapes correspond to the outer shape of an embryonic mouse limb at about 12 days of gestation).<br />
Graphic objects are stored internally as a python list in vp.actors (as vtkActor, filename or vtkPolyData):
```python
vp = Plotter()  
vp.load('data/250.vtk', c=(1,0.4,0)) # c=(R,G,B) color, name or hex code
vp.load('data/270.vtk', c=(1,0.6,0))
vp.load('data/290.vtk', c=(1,0.8,0))
print ('Loaded vtkActors: ', len(vp.actors))
vp.show()
```
![ex2](https://user-images.githubusercontent.com/32848391/32666969-90a7dc48-c639-11e7-8795-b139166f0504.png)
<br />

Draw a spline that goes through a set of points, do not show the points *(nodes=False)*:
```python
from random import gauss as g
pts = [(g(0,.1)+i/20., g(0,.1)+i/20., g(0,.1)) for i in range(100)]
vp = Plotter()  
vp.spline(pts, s=2, smooth=1.1, nodes=False)
vp.show()
```
![ex3](https://user-images.githubusercontent.com/32848391/32666970-90c1b38e-c639-11e7-92dd-336f2aa2a2cf.png)
<br />



Plot the function *f(x,y) = sin(3*x)*log(x-y)/3* (more 
examples [here](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fxy.py)).
<br />
Red dots in the plot indicate the (x,y) where the function *f* is not a real number:
```python
vp = Plotter()  
vp.fxy('sin(3*x)*log(x-y)/3', texture='paper')
vp.show()
```
![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png)
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


Draw a number of mesh objects in various formats and options:
```python
vp = Plotter(shape=(3,3), sharecam=False, interactive=0)
vp.show(at=0, c=0, actors='data/beethoven.ply', ruler=1, axes=0)
vp.show(at=1, c=1, actors='data/cow.g', wire=1)
vp.show(at=2, c=2, actors='data/limb.pcd')
vp.show(at=3, c=3, actors='data/shapes/spider.ply')
vp.show(at=4, c=4, actors='data/shuttle.obj')
vp.show(at=5, c=5, actors='data/shapes/magnolia.vtk')
vp.show(at=6, c=6, actors='data/shapes/man.vtk', axes=1)
vp.show(at=7, c=7, actors='data/teapot.xyz', axes=2)
vp.show(at=8, c=8, actors='data/pulley.vtu', axes=3)
vp.show(interactive=1)
```
![objects](https://user-images.githubusercontent.com/32848391/43654734-8d126a96-974c-11e8-80d6-73cf224c0511.png)
<br />


Cut a set of shapes with a plane that goes through the point at x=500 and has normal (0, 0.3, -1).
Wildcards are ok to load multiple files or directories:
```python
vp = Plotter()  
vp.load('data/*.vtk', c='orange', bc='aqua', alpha=1)
for a in vp.actors:
    vp.cutPlane(a, origin=(500,0,0), normal=(0,0.3,-1))
vp.show()
```
![cut](https://user-images.githubusercontent.com/32848391/33214256-6895e478-d12b-11e7-8b4b-390b698e9ef4.png)
<br />


Apply Moving Least Squares algorithm to a large point cloud to obtain a smooth surface 
from a set of scattered points in space ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)):
```bash
python examples/advanced/moving_least_squares2D.py
```
![bunnymls](https://user-images.githubusercontent.com/32848391/43954472-ef161148-9c9c-11e8-914d-1ba57718da74.png)
<br />


Simulation of a spring in a viscous medium  ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/spring.py)):
```bash
python examples/basic/spring.py
```
![spring](https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif)
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


Simulation of bacteria types that divide at different rates. 
As they divide they occupy more and more space
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cell_main.py)):
```bash
python examples/advanced/cells_main.py
```
![cells](https://user-images.githubusercontent.com/32848391/39751599-ea32aa66-52b8-11e8-93a3-4a5a65d34612.gif)
<br />


Simulation of a gyroscope hanging from a spring
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/gyroscope1.py)):
```bash
python examples/basic/gyroscope1.py
```
![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)
<br />


Simulation of [Rutherford scattering](https://en.wikipedia.org/wiki/Rutherford_scattering) 
of charged particles on a fixed target (by T. Vandermolen)
([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/particle_simulator.py)):
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


More examples can be found in directories: *examples/basic* and *examples/advanced* .

If you need to do more complicated things (define widgets.. etc), 
you still have full access to all
standard VTK objects (e.g. interactors and renderers 
through *vp.interactor, vp.renderer*... etc).<br />
In linux systems with *ffmpeg* you can use *vp.openVideo(), 
video.addFrame()* and *video.close()* to save a *movie.mp4* file.
<br />
To produce animated gifs online, check out [this great site](https://ezgif.com).

## List of available methods with default values:
```python
#
# Methods in Plotter class
# (all methods in vtkplotter.shapes and vtkplotter.analysis are also accessible)
# Example:
# from vtkplotter import Plotter
# vp = Plotter()
# vp.load('somefile.obj')
# vp.show()
def Plotter(shape=(1,1), size='auto', N=None, pos=(0,0), screensize='auto', title='vtkplotter',
            bg=(1,1,1), bg2=None, axes=0, verbose=True, interactive=True)
def load(filesOrDirs, c='gold', alpha=0.2, wire=False, bc=None, edges=False, legend=True, texture=None)
def show(actors=None, at=0, legend=None, axes=0, ruler=False, c='gold', bc=None, 
         alpha=0.2, wire=False, resetcam=True, interactive=None, q=False)
def render(addActor=None, at=None, axes=None, resetcam=False, zoom=False, rate=None) # use inside loops
def clear(actors=[])
def addActor(actor)
def removeActor(actor)
def lastActor()
def getActors(obj=None)
def moveCamera(camstart, camstop, fraction)
def cube(pt, r=1, c='g', alpha=1, legend=None, texture=None)
def light(pos, fp, deg=25, diffuse='y', ambient='r', specular='b', showsource=False)
def screenshot(filename='screenshot.png')
def write(obj, fileoutputname)
def addTrail(actor=None, maxlength=None, n=25, c=None, alpha=None, lw=1)
def addScalarBar(actor=None, c='k', horizontal=False)
def addScalarBar3D(actor=None, pos, normal=[0,0,1], sx=.1, sy=2, nlabels=9, ncols=256, cmap='jet', c='k', alpha=1)
def addSlider(sliderfunc, xmin=0, xmax=1, value=None, pos=4, title='', c='k', showValue=True)
def addButton(buttonfunc, states=['On', 'Off'], c=['w','w'], bc=['dg','dr'],
              pos=[20,40], size=24, font='arial', bold=False, italic=False, alpha=1, angle=0)
def addIcon(iconActor, pos=3, size=0.08)               
def addCutterTool(actor) 
def openVideo(name='movie.avi', fps=12, duration=None, format="XVID")
def addFrameVideo()
def pauseVideo(pause)
def closeVideo()
#
# Basic shapes creation
# Example:
# from vtkplotter.shapes import sphere
# mysphere = sphere() # returns the vtkActor
def arrow(startPoint, endPoint, s=0.03, c='r', alpha=1, legend=None, texture=None)
def arrows(startPoints, endPoints=None, c='r', s=None, alpha=1, legend=None)
def box(pos, length=1, width=2, height=3, normal=(0,0,1),c='g', alpha=1, wire=False, legend=None, texture=None)
def cone(pos, r, height, axis=[1,1,1], c='g', alpha=1, legend=None, texture=None)
def cylinder(pos, r, height, axis=[1,1,1], c='teal', alpha=1, edges=False, legend=None, texture=None, res=24)
def disc(pos, normal=[0,0,1], r1=0.5, r2=1, c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, res=12)
def ellipsoid(points, c='c', alpha=0.5, legend=None, texture=None, res=24)
def grid( pos, normal=(0,0,1), sx=1, sy=1, c='g', bc='darkgreen', lw=1, alpha=1, legend=None, resx=10, resy=10)
def helix(startPoint, endPoint, coils=12, r=1, thickness=1, c='gray', alpha=1, legend=None, texture=None)
def hyperboloid(pos, a2=1, value=0.5, height=1, axis=[0,0,1], c='m', alpha=1, legend=None, texture=None, res=50)
def line(p0, p1, lw=1, tube=False, dotted=False, c='r', alpha=1, legend=None)
def lines(plist0, plist1=None, lw=1, dotted=False, c='r', alpha=1, legend=None)   
def paraboloid(pos, r=1, height=1, axis=[0,0,1], c='cyan', alpha=1, legend=None, texture=None, res=50)
def plane(pos, normal=(0,0,1), sx=1, sy=None, c='g', bc='darkgreen', alpha=1, legend=None, texture=None)
def points(plist, c='b', tags=[], r=10, alpha=1, legend=None)
def polygon(pos, normal=(0,0,1), nsides=6, r=1, c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, followcam=0)
def pyramid(pos, s=1, height=1, axis=[0,0,1], c='dg', alpha=1, legend=None, texture=None)
def ring(pos, r=1, thickness=0.1, axis=[1,1,1], c='khaki', alpha=1, legend=None, texture=None, res=30)
def sphere(pos, r=1, c='r', alpha=1, legend=None, texture=None)
def spheres(centers, r=1, c='r', alpha=1, wire=False, legend=None, texture=None, res=8)
def text(txt, pos, normal=(0,0,1), s=1, c='k', alpha=1, bc=None, texture=None, followcam=False)
#
# Analysis methods in vtkplotter.analysis
def xyplot(points, title='', c='r', pos=1, lines=False)
def histogram(self, values, bins=10, vrange=None, title='', c='b', corner=1, lines=True)
def fxy(z='sin(x)+y', x=[0,3], y=[0,3], zlimits=[None, None], showNan=True, zlevels=10, 
        c='b', bc='aqua', alpha=1, legend=True, texture=None, res=100)
def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None)
def curvature(actor, method=1, r=1, alpha=1, lut=None, legend=None)
def boundaries(actor, c='p', lw=5, legend=None)
def extractLargestRegion(actor, c=None, alpha=None, wire=False, bc=None, edges=False, legend=None, texture=None)
def delaunay2D(actor, tol=None) # triangulate after projecting on the xy plane
def align(source, target, iters=100, legend=None):
def spline(points, smooth=0.5, degree=2, s=5, c='b', alpha=1., nodes=False, legend=None, res=20)
def fitLine(points, c='orange', lw=1, alpha=0.6, tube=False, legend=None)
def fitPlane(points, c='g', bc='darkgreen', legend=None)
def fitSphere(self, coords, c='r', alpha=1, wire=0, legend=None)
def pca(points, pvalue=.95, c='c', alpha=0.5, pcaaxes=False, legend=None)
def smoothMLS1D(actor, f=0.2, showNLines=0)
def smoothMLS2D(actor, f=0.2, decimate=1, recursive=0, showNPlanes=0)
def recoSurface(points, bins=256, c='gold', alpha=1, wire=False, bc='t', edges=False, legend=None)
def cluster(points, radius, legend=None)
def removeOutliers(points, radius, c='k', alpha=1, legend=None)
def cutPlane(actor, origin=(0,0,0), normal=(1,0,0), showcut=True, showline=False, showpts=False)
def closestPoint(surf, pt, locator=None, N=None, radius=None)
def surfaceIntersection(actor1, actor2, tol=1e-06, lw=3, c=None, alpha=1, legend=None)
def booleanOperation(actor1, actor2, operation='plus',  # possible operations: plus, intersect, minus
                     c=None, alpha=1, wire=False, bc=None, edges=False, legend=None, texture=None)
def intersectWithLine(actor, p0, p1)
#
# Methods in vtkplotter.utils
# Example:
# from vtkplotter.utils import makeActor
# makeActor(mypolydata, c='red') # returns a vtkActor
def area(actor)
def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1)
def averageSize(actor)
def cellCenters(actor)
def cellColors(actor, scalars, cmap='jet')
def cellScalars(actor, scalars, name)
def centerOfMass(actor)
def cleanPolydata(actor, tol=None)
def clone(actor, c=None, alpha=None, wire=False, bc=None, edges=False, 
          legend=None, texture=None, rebuild=True, mirror='')
def closestPoint(actor, pt, N=1, radius=None, returnIds=False)
def coordinates(actor, rebuild=True)
def decimate(actor, fraction=0.5, N=None, verbose=True, boundaries=True)
def diagonalSize(actor)
def flipNormals(actor) # N.B. input argument gets modified
def insidePoints(actor, points, invert=False, tol=1e-05)
def intersectWithLine(act, p0, p1)
def isInside(actor, point, tol=0.0001)
def isSequence(arg)
def makeActor(poly, c='gold', alpha=0.5, wire=False, bc=None, edges=False, legend=None, texture=None)
def makeAssembly(actors, legend=None)
def maxBoundSize(actor)
def mergeActors(actors, c=None, alpha=1, wire=False, bc=None, edges=False, legend=None, texture=None)
def normalize(actor): # N.B. input argument gets modified
def orientation(actor, newaxis=None, rotation=0)
def pointColors(actor, scalars, cmap='jet')
def pointIsInTriangle(p, p1,p2,p3)
def pointScalars(actor, scalars, name)
def polydata(obj, rebuild=True, index=0):
def rotate(actor, angle, axis, axis_point=[0,0,0], rad=False)
def scalars(actor, name)
def shrink(actor, fraction=0.85)   # N.B. input argument gets modified
def stretch(actor, q1, q2)
def subdivide(actor, N=1, method=0, legend=None)
def to_precision(x, p)
def volume(actor)
def xbounds(actor)
def ybounds(actor)
def zbounds(actor)
```

Useful *Plotter* attributes:
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

Additional methods of vtkActor object. They return the actor object so that can be concatenated:
```python
# Example: actor.scale(3).pos([1,2,3]).color('blue').alpha(0.5) etc..)
actor.pos()                   # set/get position vector (setters, and getters if no argument is given)
actor.addpos(v)               # add v to current actor position
actor.x()                     # set/get x component of position (same for y and z)
#
actor.rotate(angle, axis, axis_point=[0,0,0], rad=False)  # rotate actor around axis
actor.rotateX(angle, rad=False)  # rotate actor around X (or Y or Z)
actor.orientation(newaxis=None, rotation=0)  # orient actor along newaxis and rotate around its axis
#                      as specified by rotation in degrees (if newaxis=None return polydata orientation)
actor.clone(c=None, alpha=None, wire=False, bc=None, edges=False, legend=None, texture=None)
#
actor.scale()                 # set/get scaling factor of actor
actor.normalize()             # sets actor at origin and scales its average size to 1
actor.stretch(p1, p2)         # stretch actor (typically a spring, cylinder, cone) between two points
actor.updateTrail()           # if actor has a trailing line it updates it based on its current position
#
actor.shrink(fraction=0.85)                 # shrinks the polydata triangles for visualization
actor.subdivide(N=1, method=0, legend=None) # increase the nr of vertices of the surface mesh
#
actor.color(value)            # sets/gets color
actor.alpha(value)            # sets/gets opacity
#
actor.N()                     # get number of vertex points defining the surface actor
actor.polydata(rebuild=True)  # get the actor's mesh polydata including its current transformation
                              #     (if rebuild is True : get a copy in its current associated vtkTranform)
actor.coordinates()           # get a copy of vertex points coordinates (use copy=False to get references)
actor.point(i, p=None)        # set/get i-th point in actor's polydata (slow performance!)
actor isInside(p)             # check if point p is inside actor
actor insidePoints(pts, invert=False) # return the list of points (among pts) that are inside actor
actor.normals()               # get the list of normals at the vertices of the surface
actor.normalAt(i)             # get the normal at point i (slow performance!)
actor.flipNormals()           # flip all normals directions
#
actor.xbounds()               # get (xmin, xmax) of actor bounding box (same for y and z)
actor.maxBoundSize()          # get the maximum of bounds size in x y and z
actor.averageSize()           # get an average of size of surface actor as sum(p**2)/N
actor.diagonalSize()          # get the size of the diagonal of the bounding box
#
actor.centerOfMass()          # get the center of mass of actor
actor.area()                  # get the area of actor's surface
actor.volume()                # get the volume of actor
#
actor.closestPoint(p, N=1, radius=None) # get the closest N point(s) to p on actor's surface
actor.intersectWithLine(p0, p1)         # get a list of points of intersection with segment from p0 to p1
```

Some useful shortcuts available in vtkplotter (*a la vpython*):
```python
def arange(start,stop, step)  # return a range list of floats
def vector(x, y, z=0)         # return a numpy vector (2D or 3D)
def mag(v)                    # return the size of a vector or list of vectors
def mag2(v)                   # return the square of the size of a vector 
def norm(v)                   # return the versor of a vector or list of vectors
def printc(strings, c='white', bc='', hidden=False, bold=True, blink=False,
           underline=False, dim=False, invert=False, separator=' ', box= '', end='\n')
```

Available color maps from matplotlib:
```python
# Example code:
# transform a scalar value between -10.2 and 123.4 into a RGB color, using the 'jet' map
from vtkplotter import colorMap
RGBcol = colorMap(value, name='jet', vmin=-10.2, vmax=123.4)
```
![colmaps](https://user-images.githubusercontent.com/32848391/42942959-c8b50eec-8b61-11e8-930a-00dcffdca601.png)

<br />
Tested on VTK versions 6.3, 7.1, 8.1<br />
[![Downloads](https://pepy.tech/badge/vtk)](https://pepy.tech/project/vtkplotter)
