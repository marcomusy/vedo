"""Compute the (signed) distance of one mesh to another"""
from vedo import Sphere, Cube, show

s1 = Sphere().pos(10,20,30)
s2 = Cube(c='grey4').scale([2,1,1]).pos(14,20,30)

s1.distanceTo(s2, signed=False)
s1.cmap('hot').addScalarBar('Signed\nDistance')
print(s1.pointdata["Distance"].shape)  # numpy array

show(s1, s2, __doc__ , axes=1, size=(1000,500), zoom=1.5).close()
