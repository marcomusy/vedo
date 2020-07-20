"""Dijkstra algorithm to compute the graph geodesic.

Takes as input a polygonal mesh and performs
a shortest path calculation 20 times.
"""
from vedo import *

s = Sphere(r=1.02, res=200).clean(0.007).wireframe().alpha(0.02)

paths = []
for i in range(20):
    paths.append(s.geodesic(2500, i*700))
    # print(paths[-1].info['CumulativeWeights'])

show(s, Earth(), __doc__, paths, bg2='lb', viewup="z")
