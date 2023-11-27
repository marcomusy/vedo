"""Load a tetrahedral mesh and show it in different styles."""
from vedo import TetMesh, show, dataurl

# Load a tetrahedral mesh from file
tetm = TetMesh(dataurl + 'limb.vtu')
tetm.compute_cell_size()
print(tetm)

# Assign a color to each tetrahedron based on the value of "chem_0"
tetm.celldata.select('chem_0').cmap("Blues_r").add_scalarbar()

# Make a copy of tetm and shrink the tets
msh = tetm.clone().shrink(0.5).tomesh().add_scalarbar()

# Show the two meshes side by side with comments
show([(tetm, __doc__), (msh, "..shrunk tetrahedra")], N=2).close()

