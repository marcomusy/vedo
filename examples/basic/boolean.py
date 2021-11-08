"""Boolean operations with Meshes"""
from vedo import *

settings.useDepthPeeling = True

# declare the instance of the class
plt = Plotter(shape=(2, 2), interactive=0, axes=3)

# build to sphere meshes
s1 = Sphere(pos=[-0.7, 0, 0], c="red", alpha=0.5)
s2 = Sphere(pos=[0.7, 0, 0], c="green", alpha=0.5)
plt.show(s1, s2, __doc__, at=0)

# make 3 different possible operations:
b1 = s1.boolean("intersect", s2).c('magenta')
plt.show(b1, "intersect", at=1, resetcam=False)

b2 = s1.boolean("plus", s2).c("blue").wireframe(True)
plt.show(b2, "plus", at=2, resetcam=False)

b3 = s1.boolean("minus", s2).computeNormals().addScalarBar(c='white')
plt.show(b3, "minus", at=3, resetcam=False)

interactive().close()