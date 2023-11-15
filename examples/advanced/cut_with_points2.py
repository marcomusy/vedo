"""Select cells inside a point loop"""
from vedo import *

mesh = Mesh(dataurl + "dolfin_fine.vtk").lw(1)

pts = [
    [0.85382618, 0.1909104],
    [0.85585967, 0.8721275],
    [0.07500188, 0.8680605],
    [0.10143717, 0.0607675],
]

# Make a copy and cut it
cmesh = mesh.clone().cut_with_point_loop(
    pts, on="cells", include_boundary=False, invert=False,
)
cmesh.lw(1).c("tomato")

line = Line(pts, closed=True).lw(5).c("green3")

show([(mesh, line), (cmesh, line, __doc__)], N=2).close()
