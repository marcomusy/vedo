"""Generate a time sequence of 3D shapes
(from a sphere to a tetrahedron) as noisy cloud Points,
and smooth it with Moving Least Squares (smoothMLS3D).
This make a simultaneus fit in 4D (space+time).
smoothMLS3D method returns a vtkActor where points
are color coded in bins of fitted time.
Data itself can suggest a meaningful time separation
based on the spatial distribution of points.
The nr neighbours in the local 4D fitting must be specified.
"""
from vtkplotter import *

# generate uniform points on sphere (tol separates points by 2% of actor size)
cc = Sphere(res=200).clean(tol=0.02).coordinates()
txt = Text(__doc__, c="k")

a, b, noise = 0.2, 0.4, 0.1  # some random warping paramenters, and noise factor
sets = []
for i in range(5):  # generate a time sequence of 5 shapes
    cs = cc + a * i * cc ** 2 + b * i * cc ** 3  # warp sphere in weird ways
    # set absolute time of points actor, and add 1% noise on positions
    ap = Points(cs, c=i, alpha=0.5).addGaussNoise(1.0).time(0.2 * i)
    sets.append(ap)

show(sets, txt, at=0, N=2, bg="w", zoom=1.4)

sm3d = smoothMLS3D(sets, neighbours=10)

#sm3d.addScalarBar3D(pos=(-2, 0, -1))  # color indicates fitted time

show(sm3d, at=1, zoom=1.4, axes=4, interactive=1)
