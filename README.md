
# vtkplotter
A python helper class to easily draw, analyse and animate 3D objects. 
A [VTK](https://www.vtk.org/) alternative to [VPython](http://vpython.org/).

## Install and Run:
Simply type:
```bash
# Install:
(sudo) pip install vtkplotter
```

## Usage examples<br />
Download and Run the tutorials:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  # on macOS try pythonw instead
```

Simple command line usage:
```bash
vtkplotter data/*.vtk  
# other valid formats: [vtu,vts,vtp, ply,obj,stl,xml,neutral,gmsh,pcd,xyz,txt,byu, tif,slc, png,jpg]
```
<br />


From within your python script:<br />
Load a simple OBJ file and display it.<br />
Press *Esc* to close the window and exit python session or *q* to continue:
```python
from vtkplotter import Plotter

vp = Plotter()  # Declare an instance of the class
vp.show('data/shuttle.obj')
```
![shuttle](https://user-images.githubusercontent.com/32848391/35975974-e1235396-0cde-11e8-9880-69335cc7fd43.png)
<br />

Load 3 actors assigning each a different color, use their file names as legend entries.<br />
(the tridimensional shape corresponds to the outer shape of an embryonic mouse
limb at about 12 days of gestation).<br />
Graphic objects are stored internally in vp.actors (as vtkActor, filename or vtkPolyData):
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



Plot the function *f(x,y) = sin(3*x)*log(x-y)/3* (more examples in *examples/fxy.py*).
<br />
Red dots in the plot indicate the (x,y) where the function *f* is not a real number:
```python
vp = Plotter()  
vp.fxy('sin(3*x)*log(x-y)/3', texture='paper')
vp.show()
```
![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png)
<br />


Draw a bunch of basic goemetric objects on separate parts of the rendering window:
```python
vp = Plotter(N=6, sharecam=False)
vp.show( vp.arrow([0,0,0], [1,1,1]),   at=0, legend='arrow()' )
vp.show( vp.line([0,0,0], [1,1,1]),    at=1, legend='line()' )
vp.show( vp.point([1,2,3]),            at=2, legend='point()' )
vp.show( vp.text('Hello', bc=(1,0,0)), at=3 )
vp.show( vp.sphere(),                  at=4 )
vp.show( vp.cube(),                    at=5, legend='cube()')
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
![objects](https://user-images.githubusercontent.com/32848391/33093360-158b5f2c-cefd-11e7-8cb7-9e3c303b41be.png)
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


Apply Moving Least Squares algorithm to a point cloud (20k points) to obtain a smooth surface 
from a set of scattered points in space:
```bash
python examples/advanced/moving_least_squares2D.py
```
![mls](https://user-images.githubusercontent.com/32848391/40891869-dd4df456-678d-11e8-86c4-c131207868e8.png)
<br />


Motion of a large brownian
particle in a swarm of small particles in 2D motion.
The spheres collide elastically with themselves and
with the walls of the box. The masses of the spheres
are proportional to their volume.
```bash
python examples/advanced/brownian2D.py
```
![brownian](https://user-images.githubusercontent.com/32848391/36788300-b07fd4f8-1c8d-11e8-9bdd-790c6abddd99.gif)
<br />


Simulation of a spring in a viscous medium:
```bash
python examples/spring.py
```
![spring](https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif)
<br />


Motion of particles of gas in a toroidal tank. 
```bash
python examples/advanced/gas.py
```
![gas](https://user-images.githubusercontent.com/32848391/39139206-90d644ca-4721-11e8-95b9-8aceeb3ac742.gif)
<br />



Simulation of an elastic multiple pendulum with friction:
```bash
python examples/advanced/multiple_pendulum.py
```
![pendulum](https://user-images.githubusercontent.com/32848391/39259507-dc26b18a-48b6-11e8-94fd-3fcb01661b55.gif)
<br />


Direct integration of the wave equation comparing the simple Euler method (green) with the more sofisticated Runge-Kutta 4th order method (red):
```bash
python examples/advanced/wave_equation.py
```
![wave](https://user-images.githubusercontent.com/32848391/39360796-ea5f9ef0-4a1f-11e8-85cb-f3e21072c7d5.gif)
<br />


Simulation of bacteria types that divide at different rates. As they divide they occupy more and more space:
```bash
python examples/advanced/cells_main.py
```
![cells](https://user-images.githubusercontent.com/32848391/39751599-ea32aa66-52b8-11e8-93a3-4a5a65d34612.gif)
<br />


Simulation of a gyroscope hanging from a spring:
```bash
python examples/gyroscope1.py
```
![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif)
<br />



Visualizing a Turing system of reaction-diffusion between two molecules:
```bash
python examples/advanced/turing.py
```
![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)
<br />


More examples in directory *examples/* 

If you need to do more complicated things (define widgets.. etc), you can still have full access to all
standard VTK objects (e.g. interactors and renderers through *vp.interactor, vp.renderer*... etc).<br />
Use *vp.openVideo(), video.addFrame()* and *video.close()* to save a *movie.avi* file.
<br />
To produce animated gifs online, check out this great site: https://ezgif.com

## List of available methods with default values:
```python
def Plotter(shape=(1,1), size='auto', N=None, screensize=(1100,1800), title='vtkplotter',
            bg=(1,1,1), bg2=None, axes=0, verbose=True, interactive=True)
def load(filesOrDirs, c='gold', alpha=0.2, wire=False, bc=None, edges=False, legend=True, texture=None)
def show(actors=None, at=0, legend=None, axes=0, ruler=False, c='gold', bc=None, 
         alpha=0.2, wire=False, resetcam=True, interactive=None, q=False)
def clear(actors=[])
def render(resetcam=False, rate=10000)
def getActors(obj=None)
def mergeActors(actors, c=None, alpha=1, wire=False, bc=None, edges=False, legend=None, texture=None)
def moveCamera(camstart, camstop, fraction)
def light(pos, fp, deg=25, diffuse='y', ambient='r', specular='b', showsource=False)
def addActor(actor)
def removeActor(actor)
def lastActor()
def screenshot(filename='screenshot.png')
def addScalarBar(actor=None, c='k', horizontal=False)
def addScalarBar3D(actor=None, pos, normal=[0,0,1], sx=.1, sy=2, nlabels=9, ncols=256, cmap='jet', c='k', alpha=1)
def openVideo(name='movie.avi', fps=12, duration=None, format="XVID")
def addFrameVideo()
def pauseVideo(pause)
def closeVideo()
#
# Basic shapes creation
def point(pos, c='b', r=10, alpha=1, legend=None)
def points(plist, c='b', tags=[], r=10, alpha=1, legend=None)
def line(p0, p1, lw=1, tube=False, dotted=False, c='r', alpha=1, legend=None)
def lines(plist0, plist1=None, lw=1, dotted=False, c='r', alpha=1, legend=None)   
def arrow(startPoint, endPoint, s=0.03, c='r', alpha=1, legend=None, texture=None)
def arrows(startPoints, endPoints=None, c='r', s=None, alpha=1, legend=None)
def sphere(pos, r=1, c='r', alpha=1, legend=None, texture=None)
def spheres(centers, r=1, c='r', alpha=1, wire=False, legend=None, texture=None, res=8)
def cube(pt, r=1, c='g', alpha=1, legend=None, texture=None)
def helix(startPoint, endPoint, coils=12, r=1, thickness=1, c='gray', alpha=1, legend=None, texture=None)
def cylinder(pos, r, height, axis=[1,1,1], c='teal', alpha=1, edges=False, legend=None, texture=None, res=24)
def cone(pos, r, height, axis=[1,1,1], c='g', alpha=1, legend=None, texture=None)
def pyramid(pos, s=1, height=1, axis=[0,0,1], c='dg', alpha=1, legend=None, texture=None)
def ring(pos, r=1, thickness=0.1, axis=[1,1,1], c='khaki', alpha=1, legend=None, texture=None, res=30)
def ellipsoid(points, c='c', alpha=0.5, legend=None, texture=None, res=24)
def paraboloid(pos, r=1, height=1, axis=[0,0,1], c='cyan', alpha=1, legend=None, texture=None, res=50)
def hyperboloid(pos, a2=1, value=0.5, height=1, axis=[0,0,1], 
                c='magenta', alpha=1, legend=None, texture=None, res=50)
def plane(pos, normal=(0,0,1), sx=1, sy=None, c='g', bc='darkgreen', alpha=1, legend=None, texture=None)
def grid( pos, normal=(0,0,1), sx=1, sy=1, c='g', bc='darkgreen', lw=1, alpha=1, legend=None, resx=10, resy=10)
def polygon(pos, normal=(0,0,1), nsides=6, r=1, 
            c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, followcam=False):
def disc(pos, normal=[0,0,1], r1=0.5, r2=1, 
            c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, res=12)
def text(txt, pos, normal=(0,0,1), s=1, c='k', alpha=1, bc=None, texture=None, followcam=False)
#
# Analysis methods
def xyplot(points, title='', c='r', pos=1, lines=False)
def histogram(self, values, bins=10, vrange=None, title='', c='b', corner=1, lines=True)
def fxy(z='sin(x)+y', x=[0,3], y=[0,3], zlimits=[None, None], showNan=True, zlevels=10, 
        c='b', bc='aqua', alpha=1, legend=True, texture=None, res=100)
#
def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None)
def curvature(actor, method=1, r=1, alpha=1, lut=None, legend=None)
def boundaries(actor, c='p', lw=5, legend=None)
def delaunay2D(actor, tol=None) # triangulate after projecting on the xy plane
#
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

Useful methods:
```python
# Example -- vp = vtkplotter.Plotter(); vp.makeActor(myolydata, c='red')
def makeActor(poly, c='gold', alpha=0.5, wire=False, bc=None, edges=False, legend=None)
def makeAssembly(actors, legend=None)
def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1)
def polydata(actor, index=0, transformed=True)
def closestPoint(actor, point, locator=None, N=1, radius=None)
def coordinates(actor)
def cellCenters(actor)
def normals(actor)
def write(actor, outputfilename)
def colorMap(value, name='rainbow', vmin=0, vmax=1) # return the color in the scale map name
def cellColors(scalars, cmap='jet')
def pointColors(scalars, cmap='jet')
```

Additional methods of vtkActor object. They return the actor object so that can be concatenated:
```python
# Example -- actor.scale(3).pos([1,2,3]).color('blue').alpha(0.5) etc..)
actor.pos()      # set/get position vector (setters, and getters if no argument is given)
actor.addpos(v)  # add v to current actor position
actor.x()        # set/get x component of position (same for y and z)
#
actor.rotate(angle, axis, axis_point=[0,0,0], rad=False)  # rotate actor around axis
actor.rotateX(angle, rad=False)  # rotate actor around X (or Y or Z)
actor.orientation(newaxis=None, rotation=0)  # orient actor along newaxis and rotate around its axis
#                      as specified by rotation in degrees (if newaxis=None return polydata orientation)
actor.clone(c=None, alpha=None, wire=False, bc=None, edges=False, legend=None, texture=None)
#
actor.scale()          # set/get scaling factor of actor
actor.normalize()      # sets actor at origin and scales its average size to 1
actor.stretch(p1, p2): # stretch actor (typically a spring, cylinder, cone) between two points
#
actor.shrink(fraction=0.85)                 # shrinks the polydata triangles for visualization
actor.subdivide(N=1, method=0, legend=None) # increase the nr of vertices of the surface mesh
#
actor.color(value)            # sets/gets color
actor.alpha(value)            # sets/gets opacity
#
actor.N()                     # get number of vertex points defining the surface actor
actor.polydata(rebuild=True)  # get the actor's mesh polydata including its current transformation
                              # (if rebuild is True : get a copy in its current associated vtkTranform)
actor.coordinates()           # get a numpy array of all vertex points
actor.point(i, p=None)        # set/get i-th point in actor's polydata (slow performance!)
actor isInside(p)             # check if point p is inside actor
actor insidePoints(pts, invert=False) # return the list of points (among pts) that are inside actor
actor.normals()               # get the list of normals at the vertices of the surface
actor.normalAt(i)             # get the normal at point i (slow performance!)
actor.flipNormals()           # filp all normals directions
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
actor.intersectWithLine(p0, p1) # get a list of points of intersection with segment from p0 to p1
actor.cutterWidget(outputname='clipped.vtk') # invoke a cutter widget for actor
```

Some useful *numpy* shortcuts available in vtkplotter (*a la vpython*):
```python
def arange(start,stop, step)  # return a range list of floats
def vector(x, y, z=0)         # return a numpy vector (2D or 3D)
def mag(v)                    # return the size of a vector or list of vectors
def norm(v)                   # return the versor of a vector or list of vectors
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
Tested on VTK versions 5.8, 6.3, 7.1, 8.1: https://www.vtk.org
