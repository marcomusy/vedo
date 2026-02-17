"""Boolean operations between
two overlapping meshes."""
from vedo import settings, Sphere, Plotter

# Better handling of semi-transparent surfaces.
settings.use_depth_peeling = True

# 2x2 layout: inputs + 3 boolean outputs.
plt = Plotter(shape=(2, 2), interactive=False, axes=3)

s1 = Sphere(pos=[-0.7, 0, 0]).c("red5", 0.5)
s2 = Sphere(pos=[0.7, 0, 0]).c("green5", 0.5)

plt.at(0).show(s1, s2, __doc__)

# Intersection.
b1 = s1.boolean("intersect", s2).c("magenta")
plt.at(1).show(b1, "intersect", resetcam=False)

# Union.
b2 = s1.boolean("plus", s2).c("blue").wireframe(True)
plt.at(2).show(b2, "plus", resetcam=False)

# Difference.
b3 = s1.boolean("minus", s2).compute_normals().add_scalarbar(c="white")
plt.at(3).show(b3, "minus", resetcam=False)

plt.interactive().close()
