"""Draw the ellipse and the ellipsoid that contains 50% of a pointcloud,
then check how many points are inside both objects"""
from vedo import *

pts = Points(np.random.randn(1000,3))
pts.rotateZ(30).scale([1.5, 2, 0.01]).pos(2,3,0)

elli2d = pcaEllipse(  pts, pvalue=0.5)
elli3d = pcaEllipsoid(pts, pvalue=0.5)

extruded = elli2d.z(-0.1).extrude(0.2) # make an oval box

printc("Inside ellipse  :", extruded.insidePoints(pts).N(), c='b')
printc("Inside ellipsoid:", elli3d.insidePoints(pts).N(), c='b')

show(pts, elli2d, elli3d, __doc__, axes=1)

