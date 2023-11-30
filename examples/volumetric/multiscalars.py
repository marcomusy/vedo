"""A Volume can have multiple
scalars associated to each voxel"""
from vedo import dataurl, Volume, printc, show
import numpy as np

vol = Volume(dataurl+'vase.vti')
nx, ny, nz = vol.dimensions()
r0,r1 = vol.scalar_range()
vol.add_scalarbar3d('original voxel scalars')

# create a set of scalars and add it to the Volume
vol.pointdata["myscalars1"] = np.linspace(r0,r1, num=nx*ny*nz)

# create another set of scalars and add it to the Volume
vol.pointdata["myscalars2"] = np.random.randint(-100,+100, nx*ny*nz)

# make SLCImage scalars the active array (can set 0, to pick the first):
printc('Arrays in Volume are:', vol.pointdata.keys(), invert=True)
vol.pointdata.select("SLCImage")  # select the first data array as the active one

# Build the isosurface of the active scalars,
# but use testscals1 to colorize this isosurface, and then smooth it
iso1 = vol.isosurface().cmap('rainbow', 'myscalars1').smooth()
iso1.add_scalarbar3d('myscalars1')

iso2 = vol.isosurface().cmap('viridis', 'myscalars2')
iso2.add_scalarbar3d('myscalars2')

show([(vol, __doc__),
       (iso1,"Colorize isosurface using\nmyscalars1"),
       (iso2,"Colorize isosurface using\nmyscalars2"),
     ], N=3, axes=1
).close()

