# vtkPlotter
A python helper class to easily draw VTK objects

Based on VTK: https://www.vtk.org
To install:
*sudo apt-get install python-vtk*

## Example pyhton usage:
```python
import plotter
vp = plotter.vtkPlotter()
vp.help() # shows a help page
```
<br />

Load a vtk file as a vtkActor and visualize it as wireframe
with a ruler on top, no axes. Quit python when closing the window:
```python
actor = vp.loadActor('290.vtk')
actor.GetProperty().SetRepresentationToWireframe()
vp.show(actor, ruler=1, axes=0, q=1)
#same as vp.show(actors=[actor], ruler=1, axes=0, q=1)
#vp.show() picks what is automatically stored in vp.actors
EX1
```
<br />

Load 3 actors assigning color, use their paths as legend
no need to use variables, as actors are stored in vp.actors
```python
vp.loadActor('250.vtk', c=(1,0.4,0))
vp.loadActor('270.vtk', c=(1,0.6,0))
vp.loadActor('290.vtk', c=(1,0.8,0))
print 'Loaded vtkActors: ', len(vp.actors)
vp.show(legend=vp.files)
EX2
```
<br />

Draw a spline that goes through a set of points, dont show the points
```python
from random import uniform as u
pts = [(u(0,1), u(0,1), u(0,1)) for i in range(10)]
vp.make_spline(pts, s=.01, nodes=False)
vp.show()
EX3
```
<br />


Show 3 planes as a grid, add a sine plot, 
add 3 axes at bounding box origin 
```python
import numpy as np
xycoords = [(np.exp(i/10.), np.sin(i/5.)) for i in range(40)]
gr  = vp.make_xyplot( xycoords )
plx = vp.make_grid(center=(0,0.5,0.5), normal=(1,0,0), c=(1,0,0))
ply = vp.make_grid(center=(0.5,0,0.5), normal=(0,1,0), c=(0,1,0))
plz = vp.make_grid(center=(0.5,0.5,0), normal=(0,0,1), c=(0,0,1))
ax  = vp.make_axes()
vp.show(axes=0)
EX4
```
<br />

Show the vtk boundaries of a vtk surface and its normals
(ratio reduces the total nr of arrows by this factor)
```python
va = vp.loadActor('data/xavis/vtk/290.vtk', c=(1,0.1,0.1))
nv = vp.make_normals(va, ratio=5)
sbound = vp.make_boundaries(va)
vp.show(actors=[va,nv, sbound], axes=1)
EX5
```
<br />


Split window in a 49 subwindows and draw somthing in 
windows nr 12 and 38. Then open and draw on an independent window
```python
vp1 = plotter.vtkPlotter(shape=(7,7), size=(900,900))
v290 = vp1.load('data/xavis/vtk/290.vtk')
v270 = vp1.load('data/xavis/vtk/270.vtk')
vp1.interactive = False
vp1.show(at=12, polys=[v290,v270])
vp1.show(at=38, polys=[v290,v270]) 
vp2 = plotter.vtkPlotter(bg=(0.9,0.9,1))
v250 = vp2.loadActor('data/xavis/vtk/250.vtk')
v260 = vp2.loadActor('data/xavis/vtk/260.vtk')
vp2.show(actors=[v250,v260])
EX6
```
<br />


Load a surface and show its curvature based on 4 different schemes:
0-gaussian, 1-mean, 2-max, 3-min
```python
vp = plotter.vtkPlotter(shape=(1,4), size=(400,1600))
v = vp.load('data/xavis/vtk/290.vtk')
vp.interactive = False
for i in [0,1,2,3]: 
    c = vp.make_curvatures(v, ctype=i, r=1, alpha=0.8)
    vp.show(at=i, actors=[c])
vp.interact() # same as setting flag interactive=True
EX7
```
<br />


Load a surface and show its curvature based on 4 different schemes:
0-gaussian, 1-mean, 2-max, 3-min
```python
vp = plotter.vtkPlotter(shape=(1,4), size=(400,1600))
v = vp.load('data/xavis/vtk/290.vtk')
vp.interactive = False
for i in [0,1,2,3]: 
    c = vp.make_curvatures(v, ctype=i, r=1, alpha=0.8)
    vp.show(at=i, actors=[c])
vp.interact() # same as setting flag interactive=True
EX9
```
<br />


Draw a bunch of other simple objects
```python
vp = plotter.vtkPlotter(shape=(2,3), size=(800,1200))
vp.axes        = True
vp.commoncam   = False
vp.interactive = False
vp.show(at=0, actors=vp.make_arrow( [0,0,0], [1,1,1] ))
vp.show(at=1, actors=vp.make_line(  [0,0,0], [1,2,3] ))
vp.show(at=2, actors=vp.make_points( [ [0,0,0], [1,1,1], [3,1,2] ] ))
vp.show(at=3, actors=vp.make_text('hello', cam=False))
vp.show(at=4, actors=vp.make_sphere([.5,.5,.5], r=0.3))
vp.show(at=5, actors=vp.make_cube(  [.5,.5,.5], r=0.3))
vp.interact()
```
<br />


Draw a line in 3D that fits a cloud of points
also show the first set of 20 points and fit a plane
```python
for i in range(500): # draw 500 fit lines superposed
    x = np.mgrid[-2:5 :20j][:, np.newaxis] # generate 20 points
    y = np.mgrid[ 1:9 :20j][:, np.newaxis]
    z = np.mgrid[-5:3 :20j][:, np.newaxis]
    data  = np.concatenate((x, y, z), axis=1)
    data += np.random.normal(size=data.shape)*0.8 # add gauss noise
    if i==0: 
        vp.make_points(data)
        vp.make_fitplane(data)
    vp.make_fitline(data, lw=10, alpha=0.01) # fit
print vp.result['slope'] # access the last fitted slope direction
vp.show()
```
<br />


Display a tetrahedral mesh (Fenics/Dolfin format)
```python
actor = vp.loadActor('data/xavis/xml/grow/290.xml')
actor.GetProperty().SetRepresentationToWireframe()
vp.show()        
```
<br />







