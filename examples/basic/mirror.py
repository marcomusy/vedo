"""Mirror a mesh along one of the Cartesian axes"""
from vedo import dataurl, Mesh, show

myted1 = Mesh(dataurl+"teddy.vtk")

myted2 = myted1.clone(deep=False).mirror("y")
myted2.pos(0,3,0).c("green")
fp = myted2.flagpole("mirrored\nmesh").follow_camera()

show(myted1, myted2, fp, __doc__,
     axes=2, viewup="z", bg2='ly').close()
