![logo](https://user-images.githubusercontent.com/32848391/34055656-8c2e2e6c-e1d0-11e7-8d9d-728b0a535cc6.png)
<br />
<br />

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
## Example usage:<br />

Simple command line usage:
```bash
plotter data/*.vtk  # other valid formats: [vtu,vts,vtp, ply,obj,stl,xml,pcd,xyz,txt,byu]

python tutorial.py  ### run a tutorial script
```
<br />


In your python script:
```python
import plotter

vp = plotter.vtkPlotter()  # Declare an instance of the class
```
<br />

Load a vtk file as a vtkActor and visualize it in wireframe style, <br />
(the tridimensional shape corresponds to the outer shape of the embryonic mouse
limb at about 12 days of gestation).<br />
Press *Esc* to close the window and exit python session or *q* to continue:
```python
actor = vp.load('data/290.vtk', wire=1)
vp.show()
#vp.show(actor)           # overrides the content of vp.actors
#vp.show(actors=[actor])  # same as above
```
![ex1](https://user-images.githubusercontent.com/32848391/32666968-908d1bf6-c639-11e7-9201-46572a2349c2.png)
<br />

Load 3 actors assigning each a different color, use their file names as legend entries.
No need to use any variables, as actors are stored internally in vp.actors:
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


Show 3 planes as a grid, add a dummy sine plot on top left,
add 3 axes at the origin:
```python
import numpy as np
xycoords = [(np.exp(i/10.), np.sin(i/5.)) for i in range(40)]
vp = plotter.vtkPlotter()
vp.xyplot( xycoords )
vp.grid(pos=(0,0.5,0.5), normal=(1,0,0), c=(1,0,0))
vp.grid(pos=(0.5,0,0.5), normal=(0,1,0), c=(0,1,0))
vp.grid(pos=(0.5,0.5,0), normal=(0,0,1), c=(0,0,1))
vp.axes()
vp.show(axes=0)
```
![ex4](https://user-images.githubusercontent.com/32848391/32666971-90dac112-c639-11e7-96ef-ec41bdf3b7e5.png)
<br />

Show the vtk boundaries of a vtk surface and its normals<br />
(*ratio* reduces the total nr of arrows by the indicated factor):
```python
vp = plotter.vtkPlotter()
va = vp.load('data/290.vtk', c=(1,0.1,0.1))
vp.normals(va, ratio=5)
vp.boundaries(va)
vp.show()
```
![ex5](https://user-images.githubusercontent.com/32848391/32666972-90f46a5e-c639-11e7-93c3-e105322ff481.png)
<br />


Split window in a 36 subwindows and draw something in
windows nr 12 and nr 33. Then open an independent window and draw on two shapes:
```python
vp1 = plotter.vtkPlotter(shape=(6,6))
vp1.renderers[35].SetBackground(.8,.9,.9)
v270 = vp1.load('data/270.vtk')   
v290 = vp1.load('data/290.vtk')
vp1.interactive = False
vp1.show(at=12, actors=[v270,v290])
vp1.show(at=33, actors=[v270,v290])
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))
v250 = vp2.load('data/250.vtk')
v270 = vp2.load('data/270.vtk')
vp2.show()
```
![ex6](https://user-images.githubusercontent.com/32848391/32666973-910d6dc4-c639-11e7-9645-e19ffdfff3d1.png)
<br />


Load a surface and show its curvature based on 4 different schemes. All four shapes
share a common vtkCamera:<br />
*0-gaussian, 1-mean, 2-max, 3-min*
```python
vp = plotter.vtkPlotter(shape=(1,4), interactive = False)
v = vp.load('data/290.vtk')
for i in [0,1,2,3]:
    c = vp.curvature(v, method=i, r=1, alpha=0.8)
    vp.show(at=i, actors=[c])
vp.show(interactive=1) # same as setting flag interactive=True
```
![ex7](https://user-images.githubusercontent.com/32848391/32666974-912de586-c639-11e7-880a-2b377cde3615.png)
<br />


Draw a bunch of simple objects on separate parts of the rendering window:
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


Draw a number of objects in various formats and options:
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


Draw a line in 3D that fits a cloud of points,
also show the first set of 20 points and fit a plane to them:
```python
for i in range(500): # draw 500 fit lines superposed
    x = np.mgrid[-2:5 :20j][:, np.newaxis] # generate 20 points
    y = np.mgrid[ 1:9 :20j][:, np.newaxis]
    z = np.mgrid[-5:3 :20j][:, np.newaxis]
    data  = np.concatenate((x, y, z), axis=1)
    data += np.random.normal(size=data.shape)*0.8 # add gauss noise
    if i==0:
        vp.points(data)
        vp.fitplane(data)
    vp.fitline(data, lw=10, alpha=0.01) # fit
print (vp.result['slope']) # access the last fitted slope direction
vp.show()
```
![plane](https://user-images.githubusercontent.com/32848391/32667173-3ad163ec-c63a-11e7-8b3d-4a8ba047eae9.png)
<br />


Display a tetrahedral mesh (Fenics/Dolfin format). The internal verteces are displayed too:
```python
vp.load('data/290.xml.gz', wire=1)
vp.show()        
```
![ex9](https://user-images.githubusercontent.com/32848391/32666976-918480bc-c639-11e7-9749-4fd0b71523ad.png)
<br />


As a short cut, the filename (or a list of filenames) can be given in the show command directly:
```python
plotter.vtkPlotter().show('data/limb.pcd') # Point cloud (PCL file format)
```
![pcd](https://user-images.githubusercontent.com/32848391/32798156-287955b4-c974-11e7-9abf-6057dd43c5db.png)
<br />



Cut a set of shapes with a plane that goes through the point at x=500 and has normal (1, 0.3, -0.2).
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



More examples in *example.py*.<br />
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
#
def point(pos, c='b', r=10, alpha=1, legend=None)
def points(plist, c='b', r=10, alpha=1, legend=None)
def line(p0, p1, lw=1, dotted=False, c='r', alpha=1, legend=None)
def sphere(pos, r=1, c='r', alpha=1, legend=None, texture=None)
def cube(pt, r=1, c='g', alpha=1, legend=None, texture=None)
def plane(pos, normal=(0,0,1), s=10, c='g', bc='darkgreen', lw=1, alpha=1, texture=None)
def grid( pos, normal=(0,0,1), s=10, N=10, c='g', bc='darkgreen', lw=1, alpha=1, texture=None)
def polygon(pos, normal=(0,0,1), nsides=6, r=1, c='coral', bc='dg', lw=1, alpha=1, legend=None, texture=None, cam=0):
def arrow(startPoint, endPoint, c='r', alpha=1, legend=None, texture=None)
def cylinder(pos, radius, height, axis=[1,1,1], c='teal', alpha=1, legend=None, texture=None)
def octahedron(pos, s=1, axis=(0,0,1), c='g', alpha=1, wire=False, legend=None, texture=None)
def cone(pos, radius, height, axis=[1,1,1], c='g', alpha=1, legend=None, texture=None)
def ellipsoid(points, c='c', alpha=0.5, legend=None, texture=None)
def paraboloid(pos, radius=1, height=1, axis=[0,0,1], c='cyan', alpha=1, legend=None, texture=None, res=50)
def hyperboloid(pos, a2=1, value=0.5, height=1, axis=[0,0,1], c='magenta', alpha=1, legend=None, texture=None, res=50)
def helix(pos, length=2, n=6, radius=1, axis=[0,0,1], lw=1, c='grey', alpha=1, legend=None, texture=None)
def pyramid(pos, s=1, height=1, axis=[0,0,1], c='dg', alpha=1, legend=None, texture=None)
def ring(pos, radius=1, thickness=0.1, axis=[1,1,1], c='khaki', alpha=1, legend=None, texture=None)
def spline(points, s=10, c='navy', alpha=1., nodes=True, legend=None)
def bspline(points, nknots=-1, s=1, c=(0,0,0.8), alpha=1, nodes=False, legend=None)
def text(txt, pos, s=1, c='k', alpha=1, bc=None, cam=True, texture=None)
#
def xyplot(points, title='', c='r', pos=1, lines=False)
def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None)
def curvature(actor, method=1, r=1, alpha=1, lut=None, legend=None)
def subDivideSurface(actor, N=1)
def boundaries(actor, c='p', lw=5, legend=None)
#
def align(source, target, rigid=False, iters=100, legend=None):
def fitLine(points, c='orange', lw=1, alpha=0.6, tube=False, legend=None)
def fitPlane(points, c='g', bc='darkgreen', legend=None)
def pca(points, pvalue=.95, c='c', alpha=0.5, pcaaxes=False, legend=None)
def cutActor(actor, origin=(0,0,0), normal=(1,0,0), showcut=True, showline=False, showpts=False)
def closestPoint(surf, pt, locator=None, N=None, radius=None)
#
def show(actors=None, at=0, legend=None, axes=None, ruler=False, interactive=None,
         outputimage=None, c='gold', bc=None, alpha=0.2, wire=False, edges=False, resetcam=True, q=False)
def clear(actors=[])
def render(resetcam=False, rate=10000)
def addActor(actor)
def removeActor(actor)
def lastActor()
def openVideo(name='movie.avi', fps=12, duration=None, format="XVID")
def addFrameVideo()
def pauseVideo(pause)
def releaseVideo()
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
def screenshot(filename='screenshot.png')
def makePolyData(spoints, addLines=True)
def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1)
def isInside(poly, point)
def getPolyData(obj, index=0)
def closestPoint(surface, point, locator=None, N=None, radius=None)
def getCoordinates(actors)
def cutterWidget(actor, outputname='clipped.vtk')
def writeVTK(obj, fileoutput)
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
actor.shrink(fraction=0.85)  # shrinks the polydata triangles for visualizion
#
actor.visible(alpha=1)       # sets opacity
#
actor.cutterWidget()         # invoke a cutter widget for actor
actor.point(i, p=None)       # set/get i-th point in actor mesh
```


<br />
Tested on VTK versions 5.8, 6.3, 7.1, 8.0: https://www.vtk.org
