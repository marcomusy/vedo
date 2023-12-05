"""Threshold a TetMesh with a scalar array"""
from vedo import *

tetm = TetMesh(dataurl + "limb.vtu")

# Threshold the tetrahedral mesh for values in the range:
tetm.threshold(above=0.9, below=1)

tetm.celldata.select("chem_0").cmap("Accent")
tetm.add_scalarbar3d("chem_0  expression levels", c="k", italic=1)

# Make a 2D clone of the scalarbar and move it to the right:
tetm.scalarbar = tetm.scalarbar.clone2d("center-right", scale=0.2)

show(tetm, __doc__, axes=1).close()
