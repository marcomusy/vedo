"""Computes the signed distance
of one mesh from another
"""
from vedo import Sphere, Cube, show

s1 = Sphere()
s2 = Cube(pos=[1,0,0], c='white', alpha=0.4)

s1.distanceTo(s2, signed=True, negate=False)

s1.addScalarBar(title='Signed\nDistance')

# print(s1.pointdata["Distance"])

show(s1, s2, __doc__ , axes=11).close()
