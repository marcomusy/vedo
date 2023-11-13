"""Create a TetMesh from a closed surface and explode it into pieces.
Press q to make it explode."""
from vedo import Points, Mesh, TetMesh, Plotter, Text2D, dataurl, settings
import pymeshfix
import tetgen

settings.default_font = "Brachium"

n = 20000
f1 = 0.005  # control the tetras resolution
f2 = 0.15  # control the nr of seeds

# repair and tetralize the closed surface
amesh = Mesh(dataurl + "bunny.obj")
meshfix = pymeshfix.MeshFix(amesh.vertices, amesh.cells)
meshfix.repair()  # will make it manifold
repaired = Mesh(meshfix.mesh)
tet = tetgen.TetGen(repaired.vertices, repaired.cells)
tet.tetrahedralize(order=1, mindihedral=50, minratio=1.5)
tmesh = TetMesh(tet.grid)

txt = Text2D(__doc__)

# pick points on the surface and use subsample to make them uniform
seeds = Points(tmesh.cell_centers).subsample(f2)

# assign to each tetrahedron the id of the closest seed point
cids = []
for p in tmesh.cell_centers:
    cid = seeds.closest_point(p, return_point_id=True)
    cids.append(cid)
tmesh.celldata["fragment"] = cids

pieces = []
for i in range(seeds.npoints):
    tc = tmesh.clone().threshold(name="fragment", above=i - 0.1, below=i + 0.1)
    mc = tc.tomesh(fill=False).color(i)
    pieces.append(mc)

############### animate
plt = Plotter(size=(1200, 800), axes=1)
plt.show(txt, pieces)
for i in range(10):
    for pc in pieces:
        cm = pc.center_of_mass()
        pc.shift(cm / 25)
    txt.text(f"{__doc__}\n\nNr. of pieces = {seeds.npoints}")
    plt.render()
plt.interactive().close()
