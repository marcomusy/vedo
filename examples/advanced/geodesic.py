"""
Dijkstra algorithm to compute the graph geodesic.

Takes as input a polygonal mesh and performs
a shortest path calculation 20 times.
"""
from vtkplotter import *

s = Sphere(r=1.05, res=200).clean(0.005).wire().alpha(0.05)

paths = []
for i in range(20):
    paths.append(geodesic(s, 500, i * 700))
    # print(paths[-1].info['CumulativeWeights'])

doc = Text(__doc__, c="w")

show(s, Earth(lw=1), doc, paths, viewup="z")
