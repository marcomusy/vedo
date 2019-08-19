#!/usr/bin/env python
#
"""
####################################################
#
# Quick tutorial.
# Check out more examples in directories:
#	examples/basic 
#	examples/advanced
#	examples/volumetric
#	examples/simulations
#	examples/other
#
#####################################################
"""
from __future__ import division, print_function
from random import gauss, uniform as u
from vtkplotter import *

print(__doc__)


# Declare an instance of the class
vp = Plotter()

# Load a vtk file as a Actor(vtkActor) and visualize it.
# (The actual mesh corresponds to the outer shape of
# an embryonic mouse limb at about 11 days of gestation).
# Choose a tomato color for the internal surface of the mesh.
vp.load(datadir+"270.vtk").c("aqua").bc("tomato") 
vp.show()  # picks what is automatically stored in python list vp.actors
# Press Esc to close the window and exit python session, or q to continue


#########################################################################################
# Load a vtk file as a vtkActor and visualize it in wireframe style
act = vp.load(datadir+"290.vtk").wireframe()
vp.show()  # picks what is automatically stored in vp.actors
# vp.show(act)          # same: store act in vp.actors and draws act only
# wire=1 is equivalent to VTK command: act.GetProperty().SetRepresentationToWireframe()


#########################################################################################
# Load 3 actors assigning each a different color,
# by default use their file names as legend entries.
# No need to use any variables, as actors are stored internally in vp.actors:
vp = Plotter(title="3 shapes")
vp.load(datadir+"250.vtk", c=(1, 0.4, 0), alpha=0.3)
vp.load(datadir+"270.vtk", c=(1, 0.6, 0), alpha=0.3)
vp.load(datadir+"290.vtk", c=(1, 0.8, 0), alpha=0.3)
print("Loaded vtkActors: ", len(vp.actors))
vp.show()


#########################################################################################
# Draw a spline through a set of points:
vp = Plotter(title="Example of splines through random points", verbose=0)

pts = [(u(0, 2), u(0, 2), u(0, 2) + i) for i in range(8)]  # build python list of points
vp += Points(pts, r=10) # add the actor points to the internal list of actors to be shown                                 # create the vtkActor

for i in range(10):
    sp = Spline(pts, smooth=i/10, degree=2).color(i)
    sp.legend("smoothing " + str(i/10.0))
    vp += sp
vp.show(viewup="z", axes=1, interactive=1)  # show internal list of actors


#########################################################################################
# Draw a cloud of points each one with a different color
# which depends on the point position itself.
# No need to instatiate the Plotter class:
rgb = [(u(0, 255), u(0, 255), u(0, 255)) for i in range(5000)]

pts = Points(rgb, c=rgb, alpha=0.8)
show(pts, bg="w", axes=0)


#########################################################################################
# Increase the number of points in a vtk mesh using subdivide()
# and show both before and after the cure in two separate renderers defined by shape=(1,2)
vp = Plotter(shape=(1, 2), axes=False)
a1 = vp.load(datadir+"beethoven.ply", alpha=1)
coords1 = a1.getPoints()
pts1 = Points(coords1, r=4, c="g").legend("#points = " + str(len(coords1)))
vp.show(a1, pts1, at=0)

a2 = a1.clone().subdivide(method=0)  # Increasing the number of points of the mesh
coords2 = a2.getPoints()
pts2 = Points(coords2, r=1).legend("#points = " + str(len(coords2)))
vp.show(a2, pts2, at=1, interactive=True)


########################################################################################
# Draw a bunch of simple objects on separate parts of the rendering window:
# split window to best accomodate 9 renderers
vp = Plotter(N=9, title="basic shapes", axes=0, bg="white")  # split window in 9 frames
# each object can be moved independently
vp.sharecam = False
vp.show(Arrow([0, 0, 0], [1, 1, 1]),    at=0)
vp.show(Line([0, 0, 0], [1, 1, 1]),     at=1)
vp.show(Points([[0, 0, 0], [1, 1, 1]]), at=2)
vp.show(Text("Hello!", pos=(0, 0, 0)),  at=3)
vp.show(Sphere(),   at=4)
vp.show(Cube(),     at=5)
vp.show(Torus(),    at=6)
vp.show(Spring(),   at=7)
vp.show(Cylinder(), at=8, interactive=1)


########################################################################################
# Draw a bunch of objects from various mesh formats. Loading is automatic.
vp = Plotter(shape=(3, 3), bg="white")  # split window in 3 rows and 3 columns
vp.sharecam = False  # each object can be moved independently
vp.show(datadir+"beethoven.ply", at=0, axes=0)  # dont show axes, add a ruler
vp.show(datadir+"cow.byu",       at=1, zoom=1.15)  # make it 15% bigger
vp.show(datadir+"limb.pcd",      at=2)
vp.show(datadir+"ring.gmsh",     at=3)
vp.show(datadir+"images/dog.jpg",at=4)  # 2d images can be loaded the same way
vp.show(datadir+"shuttle.obj",   at=5)
vp.show(datadir+"man.vtk",       at=6, axes=2)  # show negative axes from (0, 0, 0)
vp.show(datadir+"teapot.xyz",    at=7, axes=3)  # hide negative axes
vp.show(datadir+"pulley.vtu",    at=8, interactive=True)


########################################################################################
# Draw the same object with different surface textures
# (in vtkplotter/textures, alternatibvely a jpg/png file can be specified)
vp = Plotter(shape=(3, 3), verbose=0, axes=0)
mat = ["aqua", "gold2", "metal1", "ivy", "paper", "blue", "white2", "wood3", "wood7"]
for i, mname in enumerate(mat):  # mname can be any jpeg file
    sp = vp.load(datadir+"beethoven.ply").texture(mname).legend(mname)
    vp.show(sp, at=i)
vp.show(interactive=1)


#########################################################################################
# Cut a set of shapes with a plane that goes through the
# point at x=500 and has normal (0, 0.3, -1).
# Wildcards can be used to load multiple files or entire directories:
vp = Plotter(title="Cut a surface with a plane", verbose=0)
vp.load(datadir+"2*0.vtk")
for a in vp.actors:
    a.c("orange").bc("aqua")
    a.cutWithPlane(origin=(500, 0, 0), normal=(0, 0.3, -1))
vp.show(axes=1)
