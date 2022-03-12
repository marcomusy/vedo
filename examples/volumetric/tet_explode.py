"""Segment a TetMesh with a custom scalar.
Press q to make it explode"""
from vedo import TetMesh, Plotter, Text2D, dataurl

n = 20000
f1 = 0.005  # control the tetras resolution
f2 = 0.15   # control the nr of seeds

tmesh = TetMesh(dataurl+'limb_ugrid.vtk')
surf = tmesh.tomesh(fill=False)
txt = Text2D(__doc__, font="Brachium")

# pick points on the surface and use subsample to make them uniform
seeds = surf.clone().subsample(f2).ps(10).c('black')

# assign to each tetrahedron the id of the closest seed point
cids = []
for p in tmesh.cellCenters():
	cid = seeds.closestPoint(p, returnPointId=True)
	cids.append(cid)
tmesh.celldata["fragment"] = cids
#tmesh.celldata.select("fragment")# bug, has no effect, needs name=...

pieces = []
for i in range(seeds.NPoints()):
	tc = tmesh.clone().threshold(name="fragment", above=i-0.1, below=i+0.1)
	mc = tc.tomesh(fill=False).color(i)
	pieces.append(mc)

############### animate
plt = Plotter(size=(1200,800), axes=1)
plt.show(txt, pieces)
for i in range(20):
    for pc in pieces:
        cm = pc.centerOfMass()
        pc.shift(cm/25)
    txt.text(f"{__doc__}\n\nNr. of pieces = {seeds.N()}")
    plt.render()
plt.interactive().close()



