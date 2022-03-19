#Credits:
#M. Attene. A lightweight approach to repairing digitized polygon meshes.
#The Visual Computer, 2010. (c) Springer. DOI: 10.1007/s00371-010-0416-3
#http://pymeshfix.pyvista.org
#TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator
#https://github.com/pyvista/tetgen
#
# pip install pymeshfix
# pip install tetgen
#
import pymeshfix
import tetgen
import vedo

vedo.settings.useDepthPeeling = True

amesh = vedo.Mesh(vedo.dataurl+'290.vtk')

# repairing also closes the mesh in a nice way
meshfix = pymeshfix.MeshFix(amesh.points(), amesh.faces())
meshfix.repair()
repaired = vedo.Mesh(meshfix.mesh).lineWidth(1).alpha(0.5)

# tetralize the closed surface
tet = tetgen.TetGen(repaired.points(), repaired.faces())
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
tmesh = vedo.TetMesh(tet.grid)

# save it to disk
#tmesh.write("my_tetmesh.vtk")

plt = vedo.Plotter(N=3, axes=1)
plt.at(0).show("Original mesh", amesh)
plt.at(1).show("Repaired mesh", repaired)
plt.at(2).show("Tetrahedral mesh\n(click & press shift-X)", tmesh.tomesh().shrink())
plt.interactive().close()

