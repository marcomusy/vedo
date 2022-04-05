"""Voronoi convex tiling of the plane from a set of random points"""
from vedo import Points, voronoi, show
import numpy as np

points = np.random.random((500, 2))

pts = Points(points).subsample(0.02) # impose a min distance of 2%
vor = voronoi(pts, padding=0.01)
vor.cmap('Set3', "VoronoiID", on='cells').wireframe(False)

lab = vor.labels("VoronoiID", cells=True, scale=0.01)

show(pts, vor, lab, __doc__, zoom=1.3).close()


