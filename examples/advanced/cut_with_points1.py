"""Set a loop of random points on a sphere
to cut a region of the mesh"""
from vedo import *

# This affects how colors are interpolated between points
settings.interpolate_scalars_before_mapping = True

s = Sphere()
s.color("white").alpha(0.25).backface_culling()
s.pointdata['myscalars'] = s.coordinates[:,1]
print(s)

# Pick a few points on the sphere
sv = s.points[[10, 15, 129, 165]]
pts = Points(sv).ps(12)

# Cut the loop region identified by the points
scut = s.clone().cut_with_point_loop(sv, invert=False).scale(1.01)
scut.cmap("Paired", "myscalars").alpha(1).add_scalarbar()
print(scut)

show(s, pts, scut, __doc__, axes=1, viewup="z")
