from __future__ import division, print_function
from random import uniform as u
from vedo import *


# Load a vtk file as a Mesh and visualize it.
# (The actual mesh corresponds to the outer shape of
# an embryonic mouse limb at about 11 days of gestation).
# Choose a tomato color for the internal surface of the mesh.
limb1 = load(datadir+"270.vtk").c("aqua").bc("tomato")
show(limb1)
# Press F1 to close the window and exit python session, or q to continue

# Load another file and visualize it in wireframe style
limb2 = load(datadir+"290.vtk").wireframe()
show(limb1, limb2)


#########################################################################################
# Load 3 actors assigning each a different (r,g,b) color,
# No need to use any variables here, as actors are stored internally in vp.actors:
vp = Plotter(title="Three limb shapes")
vp.load(datadir+"250.vtk").color([1, 0.4, 0]).alpha(0.3) # load and put it in Plotter
vp.load(datadir+"270.vtk").color([1, 0.6, 0]).alpha(0.3)
vp.load(datadir+"290.vtk").color([1, 0.8, 0]).alpha(0.3)
print("Loaded meshes: ", len(vp.actors))
vp.show() # picks what is stored in the python list vp.actors


#########################################################################################
# Draw a spline through a set of points:
vp = Plotter(title="Example of splines through random points")

pts = [(u(0,2), u(0,2), u(0,2) + i) for i in range(8)]  # build python list of points
vp += Points(pts, r=10) # add the Points object to the internal list of actors
vp += DashedLine(pts)   # add a dashed line through the points

for i in range(10):
    sp = Spline(pts, smooth=i/10, degree=2).color(i)
    sp.legend("smoothing " + str(i/10.0))
    vp += sp                                # add the object to Plotter
vp.show(viewup="z", axes=1, interactive=1)  # show internal list of actors


#########################################################################################
# Increase the number of vertices and triangles of a Mesh with subdivide()
# show the before and after in two separate renderers defined by shape=(1,2)
vp = Plotter(shape=(1,2), axes=0, title="L. v. Beethonven")
mesh1 = vp.load(datadir+"beethoven.ply")
pts1 = Points(mesh1, r=4, c="g").legend("#points = " + str(mesh1.N()))
vp.show(mesh1, pts1, at=0)

mesh2 = mesh1.clone().subdivide()  # Increase the number of points of the mesh
pts2 = Points(mesh2, r=2, c="r").legend("#points = " + str(mesh2.N()))
vp.show(mesh2, pts2, at=1, interactive=True)


########################################################################################
# Draw a bunch of simple objects on separate parts of the rendering window:
# split window to best accommodate 9 renderers
vp = Plotter(N=9, title="basic shapes") # split window in 9 frames
vp.sharecam = False                     # each object can be moved independently
vp.show(Arrow([0, 0, 0], [1, 1, 1]),    at=0)
vp.show(Line([0, 0, 0], [1, 1, 1]),     at=1)
vp.show(Points([[0, 0, 0], [1, 1, 1]]), at=2)
vp.show(Text("Hello!", pos=(0, 0, 0)),  at=3)
vp.show(Sphere(),                       at=4)
vp.show(Cube(),                         at=5)
vp.show(Torus(),                        at=6)
vp.show(Spring(),                       at=7)
vp.show(Cylinder(),                     at=8, interactive=True)


########################################################################################
# Draw a bunch of objects from various mesh formats.
vp = Plotter(shape=(3,3))              # split window in 3 rows and 3 columns
vp.sharecam = False                    # each object can be moved independently
vp.show(datadir+"beethoven.ply", at=0, axes=0)     # dont show axes, add a ruler
vp.show(datadir+"cow.vtk",       at=1, zoom=1.15)  # make it 15% bigger
vp.show(datadir+"limb.pcd",      at=2)
vp.show(datadir+"images/dog.jpg",at=3)             # 2d images can be loaded the same way
vp.show(datadir+"shuttle.obj",   at=4)
vp.show(datadir+"man.vtk",       at=5, axes=2)     # show negative axes from (0, 0, 0)
vp.show(datadir+"teapot.xyz",    at=6, axes=3)     # hide negative axes
vp.show(datadir+"apple.ply",     at=7, interactive=True)


########################################################################################
# Draw the same object with different surface textures
# (a jpg/png file can also be specified)
vp = Plotter(shape=(3,3))
mat = ["gold", "grid", "leather", "paper1", "water", "textile", "wood1", "wood2", "wood3"]
for i, mname in enumerate(mat):  # mname can be any jpeg file
    sp = load(datadir+"beethoven.ply").texture(mname).legend(mname)
    vp.show(sp, at=i)
vp.show(interactive=1)
