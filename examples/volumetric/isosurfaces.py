"""Interactively cut a set of isosurfaces from a volumetric dataset"""
from vedo import dataurl, show, BoxCutter, Volume

# generate an isosurface the volume for each thresholds
thresholds = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]

# isos is of type Mesh
isos = Volume(dataurl+'quadric.vti').isosurface(thresholds)

plt = show(isos, __doc__, axes=1, interactive=False)

cutter = BoxCutter(isos)
plt.add(cutter)

plt.interactive()
plt.close()

