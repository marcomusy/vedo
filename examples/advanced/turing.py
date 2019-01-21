# Scalar values are read from a file and represented on a green scale
# on a mesh as a function of time.
# The difference between one time point and the next is shown as
# a blue component.
#
from __future__ import division, print_function
from vtkplotter import vector, Plotter, ProgressBar, cylinder
import numpy as np

# Load (with numpy) an existing set of mesh points and a list 
# of scalars that represent the concentration of a substance
mesh, conc, cgradfac = np.load('data/turing_data.npy', encoding='latin1')
conc = conc/1000. # normalize concentrations read from file
nc,n = conc.shape # nc= nr. of time points, n= nr. of vertices

# Create the Plotter instance and position the camera.
# (values can be copied in the code by pressing C in the rendering window)
vp = Plotter(verbose=0, axes=0, interactive=0, size=(700,700))
vp.camera.SetPosition(962, -239, 1034)
vp.camera.SetFocalPoint(0.0, 0.0, 10.0)
vp.camera.SetViewUp(-0.693, -0.479, 0.539)

pb = ProgressBar(0,nc, c='g') # a green progress bar
for t1 in pb.range():  # for each time point
    t2 = t1+1
    if t1 == nc-1: t2=t1 # avoid index overflow with last time point
    
    vp.actors=[]       # clean up the list of actors at each iteration
    vp.add(cylinder([0,0,-15], r=260, height=10, texture='marble', res=60))
    vp.add(cylinder([0,0, 10], r=260, height=50, wire=1, c='gray', res=60))

    pts, cols = [],[]    
    for i,p in enumerate(mesh): # for each vertex in the mesh
        c1, c2 = conc[t1,i], conc[t2,i]
        cgrad = abs(c2-c1)*cgradfac     # intensity of variation
        gx, gy, gz = np.random.randn(3) # make points wiggle a bit
        pts.append(p + vector(gx/4, gy/4, gz + c1*20))
        cols.append([0., c1, cgrad])     # RGB color

    vp.points(pts, c=cols, alpha=1.0, r=6)  # points actor
    vp.points(pts, c=cols, alpha=0.1, r=30) # halos actor
    vp.camera.Azimuth(60/nc) # rotate camera by a fraction
    vp.show() # show the four new actors at each iteration
    pb.print()

vp.show(interactive=1)
