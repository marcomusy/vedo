# Thresholding and slicing a TetMesh
from vedo import TetMesh, dataurl, show

tmsh = TetMesh(dataurl+'limb_ugrid.vtk').color('Spectral')
tmsh.cmap('hot').add_scalarbar3d('chem_0  expression', c='k')

vals = [0.2, 0.3, 0.8]
isos = tmsh.isosurface(vals)

slce = tmsh.slice(normal=(1,1,1)).lighting("off").lw(1)

show([
      (tmsh, "A TetMesh"),
      (isos, "Isosurfaces for values:\n"+str(vals)),
      (slce, "Slice TetMesh with plane"),
      ], N=3, axes=1).close()
