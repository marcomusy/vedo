"""Load a tetrahedral mesh and show it in different styles."""
from vedo import *

tetm = TetMesh(dataurl+'limb_ugrid.vtk')
tetm.compute_cell_size()
print(tetm)
tetm.cmap("Blues_r", "chem_0", on="cells").add_scalarbar()

# Make a copy of tetm and shrink the tets
msh = tetm.clone().shrink(0.5).tomesh().add_scalarbar()

show([(tetm, __doc__), (msh, "..shrunk tetrahedra")], N=2).close()

