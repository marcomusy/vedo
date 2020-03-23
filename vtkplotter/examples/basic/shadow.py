"""Make a shadow of 2 meshes on the wall"""
from vtkplotter import *

a = load(datadir+"spider.ply").texture('leather')
a.normalize().rotateZ(-90).addShadow(x=-4, alpha=0.5)

s = Sphere(pos=[-0.4, 1.4, 0.3], r=0.3).addShadow(x=-4)

show(a, s, Text2D(__doc__), axes=1, viewup="z")
