"""External-tools interoperability example."""
# Credits:
# M. Attene. A lightweight approach to repairing digitized polygon meshes.
# The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
# http://pymeshfix.pyvista.org
# TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator
# https://github.com/pyvista/tetgen
#
# pip install pymeshfix
# pip install tetgen
#
import sys
import vedo
import numpy as np

try:
    import pymeshfix
except ModuleNotFoundError:
    print("Skipping example: optional dependency 'pymeshfix' is not installed.")
    print("Install with: pip install pymeshfix")
    sys.exit(0)
try:
    import tetgen
except ModuleNotFoundError:
    print("Skipping example: optional dependency 'tetgen' is not installed.")
    print("Install with: pip install tetgen")
    sys.exit(0)

amesh = vedo.Mesh(vedo.dataurl + "290.vtk")

# repairing also closes the mesh in a nice way
meshfix = pymeshfix.MeshFix(amesh.points, np.array(amesh.cells))
meshfix.repair()
repaired = vedo.Mesh(meshfix.mesh).linewidth(1)

# tetralize the closed surface
tet = tetgen.TetGen(repaired.points, repaired.cells)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
tmesh = vedo.TetMesh(tet.grid)

# save it to disk
# tmesh.write("my_tetmesh.vtu")

plt = vedo.Plotter(N=3, axes=1)
plt.at(0).show("Original mesh", amesh)
plt.at(1).show("Repaired mesh", repaired)
plt.at(2).show("Tetrahedral mesh\n(click & press shift-X)", tmesh.tomesh().shrink())
plt.interactive().close()
