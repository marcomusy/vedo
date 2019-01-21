#!/usr/bin/env python
#
from __future__ import division, print_function
from random import gauss, uniform as u
import math
from vtkplotter import *


#########################################################################################
#
# Quick tutorial.
# Check out more examples in directories 
#	examples/basic 
#	examples/advanced
#	examples/volumetric
#	examples/other
#
#########################################################################################


# Declare an instance of the class
vp = Plotter(title='first example')

# Load a vtk file as a vtkActor and visualize it.
# (The actual mesh corresponds to the outer shape of 
# an embryonic mouse limb at about 11 days of gestation).
# Choose a tomato color for the internal surface of the mesh.
vp.load('data/270.vtk', c='b', bc='tomato', alpha=1) # c=(R,G,B), letter or color name
vp.show()  # picks what is automatically stored in python list vp.actors 
# Press Esc to close the window and exit python session, or q to continue


#########################################################################################
# Load a vtk file as a vtkActor and visualize it in wireframe style
act = vp.load('data/290.vtk', wire=1) 
vp.show()               # picks what is automatically stored in vp.actors
#vp.show(act)           # same: store act in vp.actors and draws act only
#vp.show(actors=[act])  # same as above
# wire=1 is equivalent to VTK command: act.GetProperty().SetRepresentationToWireframe()


#########################################################################################
# Load 3 actors assigning each a different color, 
# by default use their file names as legend entries.
# No need to use any variables, as actors are stored internally in vp.actors:
vp = Plotter(title='3 shapes')
vp.load('data/250.vtk', c=(1,0.4,0), alpha=0.3)
vp.load('data/270.vtk', c=(1,0.6,0), alpha=0.3)
vp.load('data/290.vtk', c=(1,0.8,0), alpha=0.3)
print('Loaded vtkActors: ', len(vp.actors))
vp.show()


#########################################################################################
# Draw a spline through a set of points:
vp = Plotter(title='Example of splines through 8 random points')

pts = [ (u(0,2), u(0,2), u(0,2)+i) for i in range(8) ] # build python list of points
vp.points(pts, legend='random points')                 # create the vtkActor

for i in range(10):
    sp = spline(pts, smooth=i/10, degree=2, c=i, legend='smoothing '+str(i/10))
    vp.add(sp) # add the actor to the internal list of actors to be shown
vp.show(viewup='z', interactive=1)


#########################################################################################
# Draw a cloud of points each one with a different color
# which depends on the point position itself
vp = Plotter(title='color points')

rgb = [(u(0,255), u(0,255), u(0,255)) for i in range(5000)]

vp.points(rgb, c=rgb, alpha=0.7, legend='RGB points')
vp.show()


#########################################################################################
# Show a dummy sine plot on top left,  
# and a 3D function f(x,y) = sin(3*x)*log(x-y)/3 (more examples in basic/fxy.py)
# red points indicate where the function is not real
vp = Plotter(title='Example of a 3D function plotting', axes=2)
xycoords = [(math.exp(i/10), math.sin(i/5)) for i in range(40)]
xplt = xyplot( xycoords )
#
f = fxy( 'sin(3*x)*log(x-y)/3' )
vp.show([xplt, f], viewup='z')


#########################################################################################
# Increase the number of points in a vtk mesh using subdivide()
# and show both before and after the cure in two separate renderers defined by shape=(1,2)
vp = Plotter(shape=(1,2), axes=False)
a1 = vp.load('data/beethoven.ply', alpha=1)
coords1 = a1.coordinates()
pts1 = vp.points(coords1, r=4, c='g', legend='#points = '+str(len(coords1)))
vp.show([a1, pts1], at=0)

a2 = a1.subdivide(method=0) # Increasing the number of points of the mesh
coords2 = a2.coordinates()
pts2 = vp.points(coords2, r=1, legend='#points = '+str(len(coords2)))
vp.show([a2, pts2], at=1, interactive=True)


########################################################################################
# Draw a bunch of simple objects on separate parts of the rendering window:
# split window to best accomodate 9 renderers
vp = Plotter(N=9, title='basic shapes', axes=0) # split window in 9 frames
vp.sharecam = False                             # each object can be moved independently
vp.show(at=0, actors=arrow([0,0,0],[1,1,1]),    legend='arrow' )
vp.show(at=1, actors=line([0,0,0],[1,1,1]),     legend='line' )
vp.show(at=2, actors=points([[0,0,0],[1,1,1]]), legend='points' )
vp.show(at=3, actors=text('Hello!') )
vp.show(at=4, actors=sphere() )
vp.show(at=5, actors=cube(),     legend='cube')
vp.show(at=6, actors=torus(),    legend='torus')
vp.show(at=7, actors=helix(),    legend='helix')
vp.show(at=8, actors=cylinder(), legend='cylinder', interactive=1)


########################################################################################
# Draw a bunch of objects from various mesh formats. Loading is automatic.
vp = Plotter(shape=(3,3), title='mesh formats') # split window in 3 rows and 3 columns
vp.sharecam = False                             # each object can be moved independently
vp.show('data/beethoven.ply', at=0, c=0, axes=0)    # dont show axes, add a ruler
vp.show('data/cow.g',         at=1, c=1, zoom=1.15) # make it 15% bigger
vp.show('data/limb.pcd',      at=2, c=2)
vp.show('data/ring.gmsh',     at=3, c=3, wire=1)
vp.show('data/images/dog.jpg',at=4)              # 2d images can be loaded the same way
vp.show('data/shuttle.obj',   at=5, c=5)
vp.show('data/shapes/man.vtk',at=6, c=6, axes=2) # show negative axes from (0, 0, 0)
vp.show('data/teapot.xyz',    at=7, c=7, axes=3) # hide negative axes
vp.show('data/pulley.vtu',    at=8, c=8, interactive=1)


########################################################################################
# Draw the same object with different surface textures 
# (in vtkplotter/textures, alternatibvely a jpg/png file can be specified)
vp = Plotter(shape=(3,3), verbose=0, axes=0)
mat = ['aqua','gold2','metal1','ivy','paper','blue','white2','wood3','wood7']
for i,mname in enumerate(mat): # mname can be any jpeg file
    sp = vp.load('data/beethoven.ply', texture=mname)
    vp.show(sp, at=i, legend=mname)
vp.show(interactive=1)


#########################################################################################
# Cut a set of shapes with a plane that goes through the
# point at x=500 and has normal (0, 0.3, -1).
# Wildcards can be used to load multiple files or entire directories:
vp = Plotter(title='Cut a surface with a plane')
vp.load('data/2*0.vtk', c='orange', bc='aqua')
for a in vp.actors:
    vp.cutPlane(a, origin=(500,0,0), normal=(0,0.3,-1), showcut=True)
vp.show()

