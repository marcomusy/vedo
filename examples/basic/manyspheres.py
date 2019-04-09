"""
Example that shows how to draw very large number of 
spheres (same for Points, lines) with different colors
or different radius. Resolution (res) can be specified.
"""
# (vtk versions<8.0 might be slow)
from vtkplotter import show, Spheres, Text
from random import gauss

N = 40000

print("calculating..")
cols = range(N)  # color numbers
pts = [(gauss(0, 1), gauss(0, 2), gauss(0, 1)) for i in cols]
rads = [abs(pts[i][1]) / 10 for i in cols]  # radius=0 for y=0

# all have same radius but different colors:
s0 = Spheres(pts, c=cols, r=0.1, res=3)  # res= theta-phi resolution

# all have same color (texture) but different radius along y:
s1 = Spheres(pts, r=rads, c="lb", res=8)  # .texture('gold1')

print("..rendering spheres:", N * 2)
show(s0, at=0, N=2, axes=2, viewup=(-0.7, 0.7, 0))
show(s1, Text(__doc__), at=1, zoom=1.5, interactive=1)
