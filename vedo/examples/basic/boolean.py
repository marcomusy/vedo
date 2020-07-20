"""Boolean operations with Meshes"""
from vedo import *

# declare the instance of the class
vp = Plotter(shape=(2, 2), interactive=0, axes=3)

# build to sphere meshes
s1 = Sphere(pos=[-0.7, 0, 0], c="r", alpha=0.5)
s2 = Sphere(pos=[0.7, 0, 0], c="g", alpha=0.5)
vp.show(s1, s2, __doc__, at=0)

# make 3 different possible operations:
b1 = s1.boolean("intersect", s2).c('m').legend("intersect")
vp.show(b1, at=1, resetcam=False)

b2 = s1.boolean("plus", s2).c("b").wireframe(True).legend("plus")
vp.show(b2, at=2, resetcam=False)

b3 = s1.boolean("minus", s2).legend("minus").addScalarBar(c='w')
vp.show(b3, at=3, resetcam=False)

interactive()