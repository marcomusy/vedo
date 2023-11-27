"""Metrics of quality for
the cells of a triangular mesh
(zoom to see cell label values)"""
from vedo import dataurl, Mesh, show
from vedo.pyplot import histogram

mesh = Mesh(dataurl + "panther.stl").compute_normals().linewidth(0.1).flat()

# generate a numpy array for mesh quality
mesh.compute_quality(metric=6)
mesh.cmap("RdYlBu")

hist = histogram(mesh.celldata["Quality"], xtitle="mesh quality", ac="w")
# make it smaller and position it, use_bounds makes the cam
# ignore the object when resetting the 3d qscene
hist.scale(0.6).pos(40, -53, 0).use_bounds(False)

# add a scalar bar for the active scalars
mesh.add_scalarbar3d(c="w", title="triangle quality by min(:alpha_i )")

# create numeric labels of active scalar on top of cells
labs = mesh.labels(on="cells", precision=3, scale=0.4, font="Quikhand", c="black")

cam = dict(
    pos=(59.8, -191, 78.9),
    focal_point=(27.9, -2.94, 3.33),
    viewup=(-0.0170, 0.370, 0.929),
    distance=205,
    clipping_range=(87.8, 355),
)

show(mesh, labs, hist, __doc__, bg="bb", camera=cam, axes=11).close()
