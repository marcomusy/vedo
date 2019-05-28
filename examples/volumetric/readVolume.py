"""
Work with vtkVolume objects and surface meshes
in the same rendering window.
"""
from vtkplotter import *

# Build a vtkVolume object.
# A set of transparency values - of any length - can be passed
# to define the opacity transfer function in the range of the scalar.
#  E.g.: setting alpha=[0, 0, 0, 1, 0, 0, 0] would make visible
#  only voxels with value close to center of the range (see printed histogram).
vol = load(datadir+"embryo.slc", spacing=[1, 1, 1]) # vtkVolume
vol.c("green").alpha([0, 0, 0.4, 0.9, 0.9])

printHistogram(vol, logscale=True)

# can relocate volume in space:
# vol.scale(0.3).pos([-1,1,0]).rotate(90, axis=[0,1,1])

sph = Sphere(pos=(100, 100, 100), r=20)  # add a dummy surface

doc = Text(__doc__, c="k")

# show both vtkVolume and vtkActor
show(vol, sph, doc, axes=8, verbose=0, bg="w", zoom=1.4)
