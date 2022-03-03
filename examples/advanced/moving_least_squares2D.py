"""Use a variant of the Moving Least Squares (MLS)
algorithm to project a cloud of points to become a smooth surface.
In the second window we show the error estimated for
each point in color scale (left) or in size scale (right)."""
from vedo import *
import numpy as np
printc(__doc__, invert=1)

plt1 = Plotter(N=3, axes=1)

mesh = Mesh(dataurl+"bunny.obj").normalize().subdivide()

pts = mesh.points()
pts += np.random.randn(len(pts), 3)/20  # add noise, will not mess up the original points


#################################### smooth cloud with MLS
# build the mesh points
s0 = Points(pts, r=3).color("blue")
plt1.at(0).show(s0, "original point cloud + noise")

# project s1 points into a smooth surface of points
# The parameter f controls the size of the local regression.
mls1 = s0.clone().smoothMLS2D(f=0.5)
plt1.at(1).show(mls1, "MLS first pass, f=0.5")

# mls1 is an Assembly so unpack it to get the first object it contains
mls2 = mls1.clone().smoothMLS2D(radius=0.1)
plt1.at(2).show(mls2, "MLS second pass, radius=0.1")


#################################### draw errors
plt2 = Plotter(pos=(300, 400), N=2, axes=1)

variances = mls2.info["variances"]
vmin, vmax = np.min(variances), np.max(variances)
print("min and max of variances:", vmin, vmax)
vcols = [colorMap(v, "jet", vmin, vmax) for v in variances]  # scalars->colors

sp0 = Spheres(mls2.points(), c=vcols, r=0.02) # error as color
sp1 = Spheres(mls2.points(), c="red", r=variances/4) # error as point size

mesh.color("k").alpha(0.05).wireframe()

plt2.at(0).show(sp0, "Use color to represent variance")
plt2.at(1).show(sp1, "point size to represent variance", zoom=1.3, interactive=True)
plt2.close()
plt1.close()
