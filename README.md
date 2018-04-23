
# vtkPlotter

A python helper class to easily draw, analyse and animate tridimensional objects.
<br />A VTK alternative to Vpython.

## Download / Install:
```bash
git clone https://github.com/marcomusy/vtkPlotter.git
```

To install to a fixed location (e.g. *$HOME/software*):
```bash
mv vtkPlotter $HOME/software/
```
and add these lines to your *.bashrc*:
```bash
export PYTHONPATH=$HOME/software/vtkPlotter:$PYTHONPATH
alias plotter='$HOME/software/vtkPlotter/plotter.py'
```
## Example usage and Tutorial:<br />

Simple command line usage:
```bash
plotter data/*.vtk  
# other valid formats: [vtu,vts,vtp, ply,obj,stl,xml,pcd,xyz,txt,byu, tif,slc, png,jpg]

python tutorial.py  ### run a tutorial script (on macOS try pythonw instead)
```
<br />


In your python script:
```python
import plotter

vp = plotter.vtkPlotter()  # Declare an instance of the class
```

Load a simple OBJ file and display it.<br />
Press *Esc* to close the window and exit python session or *q* to continue:
```python
vp.show('data/shuttle.obj')
```
![shuttle](https://user-images.githubusercontent.com/32848391/35975974-e1235396-0cde-11e8-9880-69335cc7fd43.png)
<br />

Load 3 actors assigning each a different color, use their file names as legend entries.<br />
(the tridimensional shape corresponds to the outer shape of the embryonic mouse
limb at about 12 days of gestation).<br />
Graphic objects are stored internally in vp.actors (as vtkActor):
```python
vp = plotter.vtkPlotter()
vp.load('data/250.vtk', c=(1,0.4,0)) # c=(R,G,B) color, name or hex code
vp.load('data/270.vtk', c=(1,0.6,0))
vp.load('data/290.vtk', c=(1,0.8,0))
print ('Loaded vtkActors: ', len(vp.actors))
vp.show()
```
![ex2](https://user-images.githubusercontent.com/32848391/32666969-90a7dc48-c639-11e7-8795-b139166f0504.png)
<br />

Draw a spline that goes through a set of points, don't show the points *(nodes=False)*:
```python
from random import uniform as u
pts = [(u(0,1), u(0,1), u(0,1)) for i in range(10)]
vp = plotter.vtkPlotter()
vp.spline(pts, s=1.5, nodes=False)
vp.show()
```
![ex3](https://user-images.githubusercontent.com/32848391/32666970-90c1b38e-c639-11e7-92dd-336f2aa2a2cf.png)
<br />


Draw a PCA ellipsoid that contains 67% of a cloud of points:
```python
pts = [(u(0,200), u(0,200), u(0,200)) for i in range(50)]
vp = plotter.vtkPlotter()
vp.points(pts)
vp.pca(pts, pvalue=0.67, pcaAxes=True)
vp.show()
```
![pca](https://user-images.githubusercontent.com/32848391/32732169-12f82a5a-c88c-11e7-9a31-f14b100374cb.png)
<br />


Plot the function *f(x,y) = sin(3*x)*log(x-y)/3* (more examples in *examples/fxy.py*).
<br />
Red dots in the plot indicate the (x,y) where the function *f* is not a real number:
```python
vp = plotter.vtkPlotter()
vp.fxy('sin(3*x)*log(x-y)/3', texture='paper')
vp.show()
```
![fxy](https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png)
<br />


Draw a bunch of basic goemetric objects on separate parts of the rendering window:
```python
vp = plotter.vtkPlotter(N=6, interactive=0)
vp.commoncam   = False
vp.show(at=0, actors=vp.arrow(),  legend='arrow()' )
vp.show(at=1, actors=vp.line(),   legend='line()' )
vp.show(at=2, actors=vp.points(), legend='points()' )
vp.show(at=3, actors=vp.text('hello', cam=False, bc=(1,0,0) ) )
vp.show(at=4, actors=vp.sphere() )
vp.show(at=5, actors=vp.cube(),   legend='cube()')
vp.show(interactive=1)
```
![ex8](https://user-images.githubusercontent.com/32848391/32666975-91690102-c639-11e7-8f7b-ad07bd6019da.png)
<br />


Draw a number of mesh objects in various formats and options:
```python
vp = plotter.vtkPlotter(shape=(3,3))
vp.commoncam   = False
vp.interactive = False
vp.show(at=0, c=0, actors='data/beethoven.ply', ruler=1, axes=0)
vp.show(at=1, c=1, actors='data/cow.g', wire=1)
vp.show(at=2, c=2, actors='data/limb.pcd')
vp.show(at=3, c=3, actors='data/shapes/spider.ply')
vp.show(at=4, c=4, actors='data/shuttle.obj')
vp.show(at=5, c=5, actors='data/shapes/magnolia.vtk')
vp.show(at=6, c=6, actors='data/shapes/man.vtk', alpha=1, axes=1)
a = vp.getActors('man')        # retrieve actors by matching legend string
a[0].rotateX(90)               #  and rotate it by 90 degrees around x
a[0].rotateY(1.57, rad=True)   #   then by 90 degrees around y
vp.show(at=7, c=7, actors='data/teapot.xyz')
vp.show(at=8, c=8, actors='data/unstrgrid.vtu')
vp.show(interactive=1)
```
![objects](https://user-images.githubusercontent.com/32848391/33093360-158b5f2c-cefd-11e7-8cb7-9e3c303b41be.png)
<br />


Cut a set of shapes with a plane that goes through the point at x=500 and has normal (0, 0.3, -1).
Wildcards are ok to load multiple files or directories:
```python
vp = plotter.vtkPlotter()
vp.load('data/*.vtk', c='orange', bc='aqua', alpha=1)
for a in vp.actors:
    vp.cutActor(a, origin=(500,0,0), normal=(0,0.3,-1))
vp.show()
```
![cut](https://user-images.githubusercontent.com/32848391/33214256-6895e478-d12b-11e7-8b4b-390b698e9ef4.png)
<br />


Motion of a large brownian
particle in a swarm of small particles in 2D motion.
The spheres collide elastically with themselves and
with the walls of the box. The masses of the spheres
are proportional to their volume.
```bash
python examples/brownian2d.py
```
![brownian](https://user-images.githubusercontent.com/32848391/36788300-b07fd4f8-1c8d-11e8-9bdd-790c6abddd99.gif)
<br />


Motion of particles of gas in a toroidal tank. 
```bash
python examples/gas.py
```
![gas](https://user-images.githubusercontent.com/32848391/39139206-90d644ca-4721-11e8-95b9-8aceeb3ac742.gif)
<br />


Simulation of a spring in a viscous medium:
```bash
python examples/spring.py
```

![spring](https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif)


More examples in directory *examples/* 

If you need to do more complicated things (define widgets.. etc), you can still access all the
usual VTK objects like interactors and renderers through *vp.interactor, vp.renderer*... etc.<br />
Use *vp.openVideo(), vp.addFrameVideo()* and *vp.closeVideo()* to save a *movie.avi* file (needs to import cv2).
<br />

## List of available methods with default values:
```python
def help()
def __init__(shape=(1,1), size='auto', N=None, screensize=(1100,1800), title='',
             bg=(1,1,1), bg2=None, axes=True, verbose=True, interactive=True)
def load(filesOrDirs, c='gold', alpha=0.2, wire=False, bc=None, edges=False, legend=True, texture=None)
def getActors(obj=None)
def moveCamera(camstart, camstop, fraction)
def light(pos, fp, deg=25, diffuse='y', ambient='r', specular='b', showsource=False)
#
def point(pos, c='b', r=10, alpha=1, legend=None)
def points(plist, c='b', tags=[], r=10, alpha=1, legend=None)
def line(p0, p1, lw=1, tube=False, dotted=False, c='r', alpha=1, legend=None)
def sphere(pos, r=1, c='r', alpha=1, legend=None, texture=None)
def cube(pt, r=1, c='g', alpha=1, legend=None, texture=None)
def plane(pos, normal=(0,0,1), s=10, c='g', bc='dg', lw=1, alpha=1, texture=None)
def grid( pos, normal=(0,0,1), s=10, c='g', bc='dg', lw=1, alpha=1, texture=None, res=10)
def polygon(pos, normal=(0,0,1), nsides=6, r=1, 
            c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, followcam=False):
def disc(pos, normal=[0,0,1], r1=0.5, r2=1, 
            c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, res=12)
def arrow(startPoint, endPoint, s=0.03, c='r', alpha=1, legend=None, texture=None)
def helix(startPoint, endPoint, coils=12, radius=1, thickness=1, c='grey', alpha=1, legend=None, texture=None)
def cylinder(pos, radius, height, axis=[1,1,1], c='teal', alpha=1, legend=None, texture=None, res=24)
def octahedron(pos, s=1, axis=(0,0,1), c='g', alpha=1, wire=False, legend=None, texture=None)
def cone(pos, radius, height, axis=[1,1,1], c='g', alpha=1, legend=None, texture=None)
def ellipsoid(points, c='c', alpha=0.5, legend=None, texture=None, res=24)
def paraboloid(pos, radius=1, height=1, axis=[0,0,1], c='cyan', alpha=1, legend=None, texture=None, res=50)
def hyperboloid(pos, a2=1, value=0.5, height=1, axis=[0,0,1], 
                c='magenta', alpha=1, legend=None, texture=None, res=50)
def pyramid(pos, s=1, height=1, axis=[0,0,1], c='dg', alpha=1, legend=None, texture=None)
def ring(pos, radius=1, thickness=0.1, axis=[1,1,1], c='khaki', alpha=1, legend=None, texture=None, res=30)
def spline(points, smooth=0.5, degree=2, s=5, c='b', alpha=1., nodes=False, legend=None, res=20)
def text(txt, pos, s=1, c='k', alpha=1, bc=None, cam=True, texture=None)
#
def xyplot(points, title='', c='r', pos=1, lines=False)
def fxy(z='sin(x)+y', x=[0,3], y=[0,3], zlimits=[None, None], showNan=True, zlevels=10, 
        c='b', bc='aqua', alpha=1, legend=True, texture=None, res=100)
def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None)
def curvature(actor, method=1, r=1, alpha=1, lut=None, legend=None)
def subDivideMesh(actor, N=1, method=0, legend=None)
def boundaries(actor, c='p', lw=5, legend=None)
#
def align(source, target, iters=100, legend=None):
def fitLine(points, c='orange', lw=1, alpha=0.6, tube=False, legend=None)
def fitPlane(points, c='g', bc='darkgreen', legend=None)
def pca(points, pvalue=.95, c='c', alpha=0.5, pcaaxes=False, legend=None)
def cutActor(actor, origin=(0,0,0), normal=(1,0,0), showcut=True, showline=False, showpts=False)
def closestPoint(surf, pt, locator=None, N=None, radius=None)
def intersectWithLine(actor, p0, p1)
#
def show(actors=None, at=0, legend=None, axes=None, ruler=False, interactive=None,
         c='gold', bc=None, alpha=0.2, wire=False, edges=False, resetcam=True, q=False)
def clear(actors=[])
def render(resetcam=False, rate=10000)
def addActor(actor)
def removeActor(actor)
def lastActor()
def addScalarBar(actor=None, c='k', horizontal=False)
def openVideo(name='movie.avi', fps=12, duration=None, format="XVID")
def addFrameVideo()
def pauseVideo(pause)
def releaseVideo()
def screenshot(filename='screenshot.png')
```

Useful *vtkPlotter* attributes:
```python
vp = plotter.vtkPlotter() #e.g.
vp.actors       # list of vtkActors to be shown
vp.renderer     # holds current renderer
vp.renderers    # list of renderers
vp.interactor   # vtk window interactor
vp.interactive  # (true) allows to interact with renderer
vp.axes         # (true) show 3D axes
vp.camera       # current vtkCamera
vp.commoncam    # (true) share the same camera in renderers
vp.legend       # list of legend entries for each actors, can be false
vp.verbose      # verbosity
vp.result       # dictionary to store extra output information
```

Useful *plotter* functions:
```python
def makeActor(poly, c='gold', alpha=0.5, wire=False, bc=None, edges=False, legend=None)
def makeAssembly(actors, legend=None)
def makePolyData(spoints, addLines=True)
def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1)
def isInside(poly, point)
def getPolyData(obj, index=0)
def closestPoint(surface, point, locator=None, N=None, radius=None)
def getCoordinates(actors)
def cutterWidget(actor, outputname='clipped.vtk')
def write(obj, outputfilename)
```

Additional methods of vtkActor object (*a la vpython*):
```python
actor.pos()      # position vector (setters, and getters if no argument is given)
actor.addpos(v)  # add v to current actor position
actor.x()        # set/get x component of position (same for y and z)
#
actor.vel()      # set/get velocity vector
actor.vx()       # set/get x component of velocity (same for y and z)
#
actor.mass()     # set/get mass
actor.axis()     # set/get orientation axis
actor.omega()    # set/get angular velocity
actor.momentum() # get momentum vector
actor.gamma()    # get Lorentz factor
#
actor.rotate(angle, axis, rad=False)  # rotate actor around axis
actor.rotateX(angle, rad=False)       # rotate actor around X (or Y or Z)
#
actor.clone(c='gold', alpha=1, wire=False, bc=None, edges=False, legend=None, texture=None)
#
actor.normalize() # sets actor at origin and scales its average size to 1
#
actor.shrink(fraction=0.85)  # shrinks the polydata triangles for visualization
#
actor.visible(alpha=1)       # sets opacity
#
actor.cutterWidget()         # invoke a cutter widget for actor
actor.point(i, p=None)       # set/get i-th point in actor mesh
```

Some useful *numpy* shortcuts available in vtkPlotter:
```python
def arange(start,stop, step)  # return a range list of floats
def vector(x,y,z=None)        # return a numpy vector (2D or 3D)
def mag(v)                    # return the size of a vector or list of vectors
def norm(v)                   # return the versor of a vector or list of vectors
```

<br />
Tested on VTK versions 5.8, 6.3, 7.1, 8.0: https://www.vtk.org
