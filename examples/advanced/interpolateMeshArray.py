"""Interpolate the arrays of a source Mesh (RandomHills)
onto another (ellipsoid) by averaging closest point values"""
from vedo import ParametricShape, Sphere, show

# RandomHills already contains the height as a scalar defined on vertices
h = ParametricShape('RandomHills')
h.cmap('hsv', vmin=0, vmax=6)
h.addScalarBar3D(title='RandomHills height scalar value')

# interpolate such values on a completely different Mesh.
# pick N=4 closest points and assign an ave value based on shepard kernel.
s = Sphere().scale([1,1,0.5]).pos(-.1,1.5,0.3).alpha(1).lw(0.1)
s.interpolateDataFrom(h, N=4, kernel='gaussian')
s.cmap('hsv', vmin=0, vmax=6)

show(h,s, __doc__, axes=1).close()
