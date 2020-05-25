"""Visualize a TetMesh with
default ray casting..
"""
from vtkplotter import *

tetmesh = load(datadir+'limb_ugrid.vtk') # returns vtkplotter.TetMesh
tetmesh.color('jet').alphaUnit(100) # make the tets more transparent
tetmesh.addScalarBar3D()

# Build a Mesh object made of all the boundary triangles
wmesh = tetmesh.toMesh(fill=False).wireframe()

# Make a copy of tetmesh and shrink the tets
shrinked = tetmesh.clone().shrink(0.5)

# Build a Mesh object and cut it
cmesh = shrinked.toMesh(fill=True)

show([(tetmesh, __doc__),
      (wmesh, "..wireframe surface"),
      (cmesh, "..shrinked tetrahedra"),
     ], N=3, axes=1)
