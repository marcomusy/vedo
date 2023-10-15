"""Voronoi convex tiling of the plane from a set of random points"""
import numpy as np
from vedo import Points, show

# Generate a set of random points in the unit square
points = np.random.random((500, 2))

# Create a Voronoi tiling of the plane from a set of points.
pts = Points(points).subsample(0.02) # impose a min distance of 2%
vor = pts.generate_voronoi(padding=0.01)
vor.cmap('Set3', "VoronoiID", on='cells').wireframe(False)

# Create a label for each cell showing its ID
labels = vor.labels("VoronoiID", on='cells', scale=0.01, justify='center')

# Plot the objects and close the window to continue
show(pts, vor, labels, __doc__, zoom=1.3).close()


