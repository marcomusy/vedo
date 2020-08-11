"""Show Volume and Mesh objects
in the same rendering window."""
from vedo import *

# Build a vtkVolume object.
# A set of transparency values - of any length - can be passed
# to define the opacity transfer function in the range of the scalar.
#  E.g.: setting alpha=[0, 0, 0, 1, 0, 0, 0] would make visible
#  only voxels with value close to center of the range (see printed histogram).
vol = load(datadir+'embryo.slc').spacing([1,1,1]) # returns Volume(vtkVolume)
vol.color(["green", "pink", "blue"])
vol.alpha([0, 0, 0.2, 0.5, 0.9])

vol.addScalarBar3D(title='color~\dot~alpha transfer function', c='k')
vol.printHistogram(logscale=True)

# can relocate volume in space:
# vol.scale(0.3).pos([-1,1,0]).rotate(90, axis=[0,1,1])

sph = Sphere(pos=(100, 100, 100), r=20)  # add a dummy surface

# show both Volume and Mesh
show(vol, sph, __doc__, axes=1, zoom=1.2)
