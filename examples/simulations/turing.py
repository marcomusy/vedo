"""
Scalar values are read from a file and represented
on a green scale on a mesh as a function of time.
The difference between one time point and the next
is shown as a blue component.
"""
from __future__ import division, print_function
from vtkplotter import *
import numpy as np

settings.renderPointsAsSpheres = False

doc = Text(__doc__, c="k")

# Load (with numpy) an existing set of mesh points and a list
# of scalars that represent the concentration of a substance
mesh, conc, cgradfac = np.load(datadir+"turing_data.npy",
                               encoding="latin1", allow_pickle=True)
conc = conc / 1000.0  # normalize concentrations read from file
nc, n = conc.shape  # nc= nr. of time Points, n= nr. of vertices

# Create the Plotter instance and position the camera.
# (values can be copied in the code by pressing C in the rendering window)
vp = Plotter(verbose=0, axes=0, interactive=0, size=(700, 700), bg="w")
#
#vp.camera.SetPosition(962, -239, 1034)
#vp.camera.SetFocalPoint(0.0, 0.0, 10.0)
#vp.camera.SetViewUp(-0.693, -0.479, 0.539)

pb = ProgressBar(0, nc, c="g")  # a green progress bar
for t1 in pb.range():  # for each time point
    t2 = t1 + 1
    if t1 == nc - 1:
        t2 = t1  # avoid index overflow with last time point

    vp.actors = [doc]  # clean up the list of actors at each iteration
    vp += Cylinder([0, 0, -15], r=260, height=10, res=60).texture("marble")
    vp += Cylinder([0, 0, 10], r=260, height=50, c="gray", res=60).wireframe(1)

    pts, cols = [], []
    for i, p in enumerate(mesh):  # for each vertex in the mesh
        c1, c2 = conc[t1, i], conc[t2, i]
        cgrad = abs(c2 - c1) * cgradfac  # intensity of variation
        gx, gy, gz = np.random.randn(3)  # make points wiggle a bit
        pts.append(p + vector(gx / 4, gy / 4, gz + c1 * 20))
        cols.append([0.0, c1, cgrad])  # RGB color

    vp += Points(pts, c=cols, alpha=1.0, r=6)   # points actor
    vp += Points(pts, c=cols, alpha=0.1, r=30)  # halos actor
    vp.show()  # show the four new actors at each iteration
    vp.camera.Azimuth(10 / nc)  # rotate camera by a fraction
    pb.print()

vp.show(interactive=1)
