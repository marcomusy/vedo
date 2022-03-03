"""Boolean operations with Meshes"""
from vedo import *

settings.useDepthPeeling = True

# declare the instance of the class
plt = Plotter(shape=(2, 2), interactive=False, axes=3)

# build to sphere meshes
s1 = Sphere(pos=[-0.7, 0, 0], c="red5", alpha=0.5)
s2 = Sphere(pos=[0.7, 0, 0], c="green5", alpha=0.5)
plt.at(0).show(s1, s2, __doc__)

# make 3 different possible operations:
b1 = s1.boolean("intersect", s2).c('magenta')
plt.at(1).show(b1, "intersect", resetcam=False)

b2 = s1.boolean("plus", s2).c("blue").wireframe(True)
plt.at(2).show(b2, "plus", resetcam=False)

b3 = s1.boolean("minus", s2).computeNormals().addScalarBar(c='white')
plt.at(3).show(b3, "minus", resetcam=False)

plt.interactive().close()
