"""Project a shadow of a mesh in a specified direction"""
from vedo import *

settings.use_depth_peeling = False # depending on your system

msh = Mesh(dataurl+"man.vtk").c("k5")

plane = Plane(pos=(0,0,-1.6), normal=(0,0,1), s=[6,7]).alpha(0.2)

shad = msh.clone().project_on_plane(plane, direction=(0.5,1,-1))
shad.c("k7").alpha(1).lighting("off").use_bounds(False)

plane.shift(0,-0,0.001) # a small tolerance to avoid coplanarity with shad

show(msh, plane, shad, __doc__, viewup='z', axes=7).close()
