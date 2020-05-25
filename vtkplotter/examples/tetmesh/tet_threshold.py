"""Threshold the original TetMesh
with a scalar array"""
from vtkplotter import *

tetmesh = load(datadir+'limb_ugrid.vtk')
tetmesh.color('prism').alpha([0,1])#.printInfo()

# Threshold the tetrahedral mesh for values in the range:
tetmesh.threshold('chem_0', above=0.9, below=1)
tetmesh.addScalarBar3D(title='chem_0  expression levels', c='k', italic=1)

show([(tetmesh,__doc__),
       tetmesh.toMesh(shrink=0.9),
     ], N=2, axes=1)
