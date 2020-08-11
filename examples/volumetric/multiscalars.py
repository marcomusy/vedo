"""A Volume can have multiple
scalars associated to each voxel"""
from vedo import *
import numpy as np

vol = load(datadir+'vase.vti')
nx, ny, nz = vol.dimensions()
r0,r1 = vol.scalarRange()
vol.addScalarBar3D(title='original voxel scalars')

# create a set of scalars and add it to the Volume
sc1 = np.linspace(r0,r1, num=nx*ny*nz)#.astype(np.uint8)
vol.addPointArray(sc1, "myscalars1")

# create another set of scalars and add it to the Volume
sc2 = np.random.randint(r0,r1, nx*ny*nz)#.astype(np.uint8)
vol.addPointArray(sc2, "myscalars2")

# make SLCImage scalars the active array (can set 0, to pick the first):
printc('Arrays in Volume are:\n', vol.getArrayNames(), invert=1)
vol.getPointArray('SLCImage')

# Build the isosurface of the active scalars,
# but use testscals1 to colorize this isosurface, and then smooth it
iso1 = vol.isosurface().cmap('jet', 'myscalars1').smoothWSinc().lw(0.1)
iso1.addScalarBar3D(title='myscalars1')

iso2 = vol.isosurface().cmap('viridis', 'myscalars2')
iso2.addScalarBar3D(title='myscalars2')

show([(vol, __doc__),
       (iso1,"Colorize isosurface using\nmyscalars1"),
       (iso2,"Colorize isosurface using\nmyscalars2"),
     ], N=3, axes=1)