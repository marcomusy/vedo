"""Draw the ellipse (dark) and the ellipsoid (light) 
that in both cases contain 50% of a point cloud,
then check how many points are inside both objects"""
from vedo import *

pts = Points(np.random.randn(1000,3))
pts.scale([2, 1.5, 0.01]).rotate_z(30).pos([50,60,0])

elli2d = pca_ellipse(  pts, pvalue=0.5)
elli3d = pca_ellipsoid(pts, pvalue=0.5).alpha(0.1)

extruded = elli2d.z(-0.1).extrude(0.2)  # make an oval box

printc("Inside ellipse  :", extruded.inside_points(pts).npoints, c='b')
printc("Inside ellipsoid:", elli3d.inside_points(pts).npoints, c='b')

show(pts, elli2d, elli3d, __doc__, axes=1, zoom='tight').close()

