"""Make a shadow of 2 meshes on the wall"""
from vedo import *

a = Mesh(dataurl+"spider.ply").texture('leather')
a.normalize().rotateZ(-90).addShadow(x=-3, alpha=0.5)

s = Sphere(r=0.3).pos(0.4,0,0.6).addShadow(x=-3)

show(a, s, __doc__, axes=1, viewup="z").close()
