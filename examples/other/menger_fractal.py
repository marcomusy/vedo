"""
Menger fractal sponge
"""
# Credits: K3D authors at
# https://github.com/K3D-tools/K3D-jupyter/tree/master/examples
#https://en.wikipedia.org/wiki/Menger_sponge
import numpy as np
from vtkplotter import Volume, legosurface, show

iteration = 4
size = 3**iteration

voxels = np.ones((size, size, size));

def iterate(length, x, y, z):

    nl = length // 3
    if nl < 1: return

    margin = (nl-1) // 2
    voxels[z-margin:z+margin+1, y-margin:y+margin+1, :] = 0
    voxels[z-margin:z+margin+1, :, x-margin:x+margin+1] = 0
    voxels[:, y-margin:y+margin+1, x-margin:x+margin+1] = 0

    for ix,iy,iz in np.ndindex((3,3,3)):
        if (1 if ix !=1 else 0) + (1 if iy != 1 else 0) + (1 if iz != 1 else 0) !=2:
            iterate(nl, x + (ix-1) * nl, y + (iy-1) * nl , z + (iz-1) * nl)

iterate(size, size//2, size//2, size//2)
print('voxels min, max =', np.min(voxels), np.max(voxels))

vol = Volume(voxels)
lego = vol.legosurface(-0.1, 1.1, cmap='afmhot_r')

show(vol, lego, N=2, bg='w')
