"""Histogram in 3D.
The size of each cube is proportional to the scalar value at that point"""
import numpy as np
from vedo import Volume, Cube, Glyph, show

X, Y, Z = np.mgrid[:4, :8, :8]
scalar_field = 2-((X-4)**2 + (Y-4)**2 + (Z-4)**2)/25

pcloud = Volume(scalar_field).topoints()

marker = Cube().scale(.4)
glyphed_pcl = Glyph(pcloud, marker, scaleByScalar=True)
glyphed_pcl.cmap('seismic').addScalarBar()

show(glyphed_pcl, __doc__, axes=1)
