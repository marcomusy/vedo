import numpy as np
from vedo import *
from vedo.pyplot import histogram, plot

cmap = 'nipy_spectral'
alpha = np.array([0, 0, 0.05, 0.2, 0.8, 1])

vol = Volume(dataurl+"embryo.slc")
vol.cmap(cmap).alpha(alpha).addScalarBar3D(c='w')
xvals = np.linspace(*vol.scalarRange(), len(alpha))

p  = histogram(vol, logscale=True, c=cmap, bc='white')
p += plot(xvals, alpha * p.ybounds()[1], '--ow').z(1)

show([
      (vol, Axes(vol, c='w'), f"Original Volume\ncolor map: {cmap}"),
      (p, "Voxel scalar histogram\nand opacity transfer function")
     ],
     N=2, sharecam=False, bg=(82,87,110),
).close()
