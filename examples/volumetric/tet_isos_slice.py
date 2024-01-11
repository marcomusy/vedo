# Thresholding and slicing a TetMesh
from vedo import TetMesh, dataurl, show

tmsh = TetMesh(dataurl+'limb.vtu')

tmsh.celldata.select('chem_0').cmap('hot')
tmsh.add_scalarbar3d('chem_0  expression', c='k')

vals = [0.2, 0.3, 0.8]
tmsh.map_cells_to_points(["chem_0"])
isos = tmsh.pointdata.select("chem_0").isosurface(vals).flat()

slce = tmsh.slice(normal=(1,1,1)).lighting("off").lw(1)

print(tmsh)
show([
      (tmsh, "A TetMesh"),
      (isos, "Isosurfaces for values:\n"+str(vals)),
      (slce, "Slice TetMesh with plane"),
      ], N=3, axes=1).close()
