"""Computes the (signed) distance
from one mesh to another.
"""
from vtkplotter import Sphere, Cube, show, Text

s1 = Sphere()
s2 = Cube(pos=[1,0,0], c='white', alpha=0.4)

s1.distanceToMesh(s2, signed=True, negate=False)

s1.addScalarBar(title='Signed\nDistance', c='w')

#print(s1.scalars("Distance"))

show(s1, s2, Text(__doc__))
