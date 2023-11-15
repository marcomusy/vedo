"""Tetralize a closed surface mesh
Click on the mesh and press ↓ or x to toggle a piece"""
from vedo import dataurl, Sphere, TessellatedBox, settings, Mesh, show

settings.use_depth_peeling = True

# surf = Sphere(quads=True, res=15)
# surf = TessellatedBox()
# surf = Mesh(dataurl+'290_closed.vtk')
surf = Mesh(dataurl+'bunny.obj', c='g3').fill_holes().cap().smooth()

tmesh = surf.tetralize(side=0.015, debug=True)
#tmesh.write('mytetmesh.vtk')  # save to disk!

# Assign an id to each tetrahedron to visualize regions
seeds = surf.clone().subsample(0.3)
cids = []
for p in tmesh.cell_centers:
	cid = seeds.closest_point(p, return_point_id=True)
	cids.append(cid)
tmesh.celldata["fragments"] = cids

pieces = []
for i in range(seeds.npoints):
	tc = tmesh.clone().threshold("fragments", above=i-0.1, below=i+0.1)
	mc = tc.shrink(0.95).tomesh().color(i)
	pieces.append(mc)

show(__doc__, pieces, axes=1)
