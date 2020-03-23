"""Computes the (signed) distance
from one mesh to another.
"""
from vtkplotter import Sphere, Cube, show, Text2D

s1 = Sphere()
s2 = Cube(pos=[1,0,0], c='white', alpha=0.4)

s1.distanceToMesh(s2, signed=True, negate=False)

s1.addScalarBar(title='Signed\nDistance')

#print(s1.getPointArray("Distance"))

show(s1, s2, Text2D(__doc__))
