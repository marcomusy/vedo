"""
Create a 3D Delaunay triangulation of input points.
"""
from vtkplotter import *

spid = load("data/shapes/spider.ply", c="brown")

ch = convexHull(spid.coordinates()).alpha(0.2)

show([spid, ch, Text(__doc__)], axes=1)
