"""Mirror a mesh along one of the Cartesian axes"""
from vedo import dataurl, Mesh, show

# myted1 = Mesh(dataurl+"teddy.vtk")

# myted2 = myted1.clone()
# myted2.pos(0,3,0).mirror("y")
# myted2.c("green")

# fp = myted2.flagpole("mirrored\nmesh").follow_camera()

# show(myted1, myted2, fp, __doc__,  bg2='ly', axes=1)


from vedo import *
# import vtk
s = Mesh(dataurl+"cessna.vtk")
s.rotate_z(30).shift(3,1)
s.mirror('xy', origin=(0,0,0))
show(s, Point(), axes=1)
exit()

s.scale([-1,1,1])

rs = vtk.vtkReverseSense()
rs.SetInputData(s)
rs.ReverseNormalsOff()
rs.Update()
outpoly = rs.GetOutput()

s.DeepCopy(outpoly)

show(s, axes=1)
exit()
# myted2 = myted1.clone()
# myted2.mirror("y")
# myted2.c("green")

# fp = myted2.flagpole("mirrored\nmesh").follow_camera()

show(myted1, myted2, axes=1, viewup='z')

# myted2.pos(0,3,0).mirror("xy")
