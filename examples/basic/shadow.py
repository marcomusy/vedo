"""Make a shadow of 2 meshes on the wall."""
from vtkplotter import *

a = load(datadir + "spider.ply").normalize().rotateZ(-90)
s = Sphere(pos=[-0.4, 1.5, 0.3], r=0.3)

shad = Shadow(a, s, direction="x").x(-4)

show(a, s, shad, Text(__doc__), axes=1, viewup="z", bg="w")
