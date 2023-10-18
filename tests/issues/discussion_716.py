
from vedo import *

msh = Polygon(nsides=5)

pts = utils.pack_spheres(msh, radius=0.1)

# optionally add some noise:
jitter = np.random.randn(len(pts),3)/1000
jitter[:,2] = 0
pts += jitter

pts = Points(pts)
pts.cut_with_line(msh.vertices) # needs an ordered set of points

show(msh, pts, axes=1)

