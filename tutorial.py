#!/usr/bin/env python
#
from __future__ import division, print_function
from random import uniform as u, gauss
import plotter
import math
  
#########################################################################################
#
# Check for more examples in directory examples/
#
#########################################################################################
# Declare an instance of the class
vp = plotter.vtkPlotter(title='first example')
# vp.help() # shows a help message

# Load a vtk file as a vtkActor and visualize it.
# The tridimensional shape corresponds to the outer shape of the embryonic mouse limb
# at about 11 days of gestation.
# Choose a tomato color for the internal surface, and no transparency.
vp.load('data/270.vtk', c='b', bc='tomato', alpha=1) # c=(R,G,B), #hex, letter or name
vp.show()             # picks what is automatically stored in vp.actors list
# Press Esc to close the window and exit python session, or q to continue


#########################################################################################
# Load a vtk file as a vtkActor and visualize it in wireframe style.
act = vp.load('data/290.vtk', wire=1) 
vp.axes = False         # do not draw cartesian axes
vp.show()               # picks what is automatically stored in vp.actors
#vp.show(act)           # same: store act in vp.actors and draws act only
#vp.show(actors=[act])  # same as above
# wire=1, equivalent to act.GetProperty().SetRepresentationToWireframe()


#########################################################################################
# Load 3 actors assigning each a different color, 
# by default use their file names as legend entries.
# No need to use any variables, as actors are stored internally in vp.actors:
vp = plotter.vtkPlotter(title='3 shapes')
vp.load('data/250.vtk', c=(1,0.4,0), alpha=.3)
vp.load('data/270.vtk', c=(1,0.6,0), alpha=.3)
vp.load('data/290.vtk', c=(1,0.8,0), alpha=.3)
print ('Loaded vtkActors: ', len(vp.actors))
vp.show()


#########################################################################################
# Draw a spline that goes through a set of points, show the points (nodes=True):
vp = plotter.vtkPlotter(title='Example of splines through 8 random points')

pts = [ (u(0,2), u(0,2), u(0,2)+i) for i in range(8) ]

vp.points(pts, legend='random points')

for i in range(10):
    vp.spline(pts, smooth=i/10, degree=2, c=i, legend='spline #'+str(i))
vp.show()


#########################################################################################
# Draw the PCA ellipsoid that contains 50% of a cloud of points,
# check if points are inside the actor surface:
vp = plotter.vtkPlotter(title='Example of PCA analysys')
pts = [(gauss(0,1), gauss(0,2), gauss(0,3)) for i in range(1000)]
a = vp.pca(pts, pvalue=0.5, pcaAxes=1, legend='PCA ellipsoid')

ipts = vp.insidePoints(a, pts)
opts = vp.insidePoints(a, pts, invert=True)
vp.points(ipts, c='g', legend='in  points #'+str(len(ipts)))
vp.points(opts, c='r', legend='out points #'+str(len(opts)))
vp.show()


#########################################################################################
# Draw a cloud of points each one with a different color
# which depends on the point position itself
vp = plotter.vtkPlotter(title='color points')

rgb = [(u(0,255), u(0,255), u(0,255)) for i in range(2000)]

vp.points(rgb, c=rgb, alpha=0.7, legend='RGB points')
vp.show()


#########################################################################################
# Show a dummy sine plot on top left,  
# and a 3D function f(x,y) = sin(3*x)*log(x-y)/3 (more examples in examples/fxy.py)
# red points indicate where the function is not real
vp = plotter.vtkPlotter(title='Example of a 3D function plotting', axes=2)
xycoords = [(math.exp(i/10.), math.sin(i/5.)) for i in range(40)]
vp.xyplot( xycoords )

vp.fxy( 'sin(3*x)*log(x-y)/3', texture='paper' )
vp.show()


#########################################################################################
# Show the vtk boundaries of a vtk surface and its normals
# (ratio reduces the total nr of arrows by the indicated factor):
vp = plotter.vtkPlotter(title='Normals and surface edges')
va = vp.load('data/290.vtk', c='maroon', legend=0)
vp.normals(va, ratio=5, legend=False)
vp.boundaries(va)
vp.show(legend='shape w/ edges')


#########################################################################################
# Increases the number of points in a vtk mesh using subdivide()
# and shows them before and after.
vp = plotter.vtkPlotter(shape=(1,2), axes=False)
a1 = vp.load('data/beethoven.ply', alpha=1)
coords1 = a1.coordinates()
pts1 = vp.points(coords1, r=4, c='g', legend='#points = '+str(len(coords1)))
vp.show([a1, pts1], at=0, interactive=False)

a2 = a1.subdivide(method=0) # Increasing the number of points of the mesh
coords2 = a2.coordinates()
pts2 = vp.points(coords2, r=1, legend='#points = '+str(len(coords2)))
vp.show([a2, pts2], at=1, interactive=True)


#########################################################################################
# Load a surface and show its curvature based on 4 different schemes.
# All four shapes share a common vtkCamera:
# 0-gaussian, 1-mean, 2-max, 3-min
vp = plotter.vtkPlotter(shape=(1,4), title='surface curvature')
v = vp.load('data/290.vtk')
vp.interactive = False
vp.axes = False
for i in [0,1,2,3]:
    c = vp.curvature(v, method=i, r=1, alpha=0.8)
    vp.show(c, at=i, legend='method #'+str(i+1))
vp.show(interactive=1)


########################################################################################
# Draw a simple objects on separate parts of the rendering window:
# split window to best accomodate 6 renderers
vp = plotter.vtkPlotter(N=9, title='basic shapes')
vp.sharecam   = False
vp.interactive = False
vp.show(at=0, actors=vp.arrow([0,0,0],[1,1,1]),  legend='arrow()' )
vp.show(at=1, actors=vp.line([0,0,0],[1,1,1]),   legend='line()' )
vp.show(at=2, actors=vp.points([[0,0,0],[1,1,1]]), legend='points()' )
vp.show(at=3, actors=vp.text('Hello', bc='r', followcam=False) )
vp.show(at=4, actors=vp.sphere([0,0,0], r=1) )
vp.show(at=5, actors=vp.cube(),     legend='cube()')
vp.show(at=6, actors=vp.ring(),     legend='ring()')
vp.show(at=7, actors=vp.helix(),    legend='helix()')
vp.show(at=8, actors=vp.cylinder(), legend='cylinder()')
vp.show(interactive=1)


########################################################################################
# Draw a bunch of objects in many formats. Split window in 3 rows and 3 columns
vp = plotter.vtkPlotter(shape=(3,3), title='Example 12', interactive=False)
vp.sharecam   = False
vp.show(at=0, c=0, actors='data/beethoven.ply', ruler=1, axes=0)
vp.show(at=1, c=1, actors='data/cow.g', wire=1)
vp.show(at=2, c=2, actors='data/limb.pcd')
vp.show(at=3, c=3, actors='data/shapes/spider.ply')
vp.show(at=4, c=4, actors='data/shuttle.obj')
vp.show(at=5, c=5, actors='data/shapes/magnolia.vtk')
vp.show(at=6, c=6, actors='data/shapes/man.vtk', alpha=1, axes=1)
vp.show(at=7, c=7, actors='data/teapot.xyz')
vp.show(at=8, c=8, actors='data/unstrgrid.vtu')
vp.show(interactive=1)
a = vp.getActors('man')         # retrieve all actors with matching legend string
a[0].rotateX(-90)               #  and rotate the first by 90 degrees around x
a[0].rotateY(-1.57, rad=True)   #  and then by 90 degrees around y
vp.show()


########################################################################################
# Draw the same object with different surface textures
vp = plotter.vtkPlotter(shape=(3,3), verbose=0, axes=0, interactive=0)
mat = ['aqua','gold2','metal1','ivy','paper','blue','white2','wood3','wood7']
for i,mname in enumerate(mat): # mname can be any jpeg file
    sp = vp.load('data/beethoven.ply', alpha=1, texture=mname)
    vp.show(sp, at=i, legend=mname)
vp.show(interactive=1)


#########################################################################################
# Cut a set of shapes with a plane that goes through the
# point at x=500 and has normal (0, 0.3, -1).
# Wildcards are ok to load multiple files or directories:
vp = plotter.vtkPlotter(title='Cut a surface with a plane')
vp.load('data/*.vtk', c='orange', bc='aqua')
for a in vp.actors:
    vp.cutPlane(a, origin=(500,0,0), normal=(0,0.3,-1))
vp.show()

  
#########################################################################################
# Find closest point in set pts1 to pts2 within a specified radius
vp = plotter.vtkPlotter(title='closest points example')

pts1 = [(u(0,5), u(0,5), u(0,5)) for i in range(40)]
pts2 = [(u(0,5), u(0,5), u(0,5)) for i in range(20)]

vp.points(pts1, r=4,  alpha=1, legend='point set 1')
vp.points(pts1, r=25, alpha=0.1) # make a halo

a = vp.points(pts2, r=4, c='r', alpha=1, legend='point set 2')

#for each point in pts1 find the points within radius=2 
#and pick one (not necessarily the closest)
for p in pts1:
    pts = a.closestPoint(p, radius=2)
    if len(pts): vp.line(p, pts[0])
vp.show()


#########################################################################################
a = vp.load('data/cow.g')
a.cutterWidget() # invoke cutter widget

