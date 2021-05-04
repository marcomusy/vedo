# Thresholding and slicing a TetMesh
from vedo import TetMesh, dataurl, show

tetmesh = TetMesh(dataurl+'limb_ugrid.vtk').color('Spectral')
tetmesh.addScalarBar3D(title='chem_0  expression', c='k')

thrslist = [0.2, 0.3, 0.8]
isos = tetmesh.isosurface(thrslist)

slce = tetmesh.slice(normal=(1,1,1)).lw(0.1)

show([
      (tetmesh, "A TetMesh"),
      (isos, "Isosurfaces for thresholds:\n"+str(thrslist)),
      (slce, "Slice TetMesh with plane"),
     ], N=3, axes=1, viewup='z').close()
