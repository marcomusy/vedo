"""Find the overlap area of 2 triangles"""
from vedo import Mesh, precision, show
import numpy as np

verts1 = [(1.9, 0.50), (2.1, 0.8), (2.4, 0.4)]
verts2 = [(2.3, 0.75), (1.7, 0.4), (2.1, 0.3)]
faces  = [(0,1,2)]

m1 = Mesh([verts1, faces]).c('g').lw(4).wireframe()
m2 = Mesh([verts2, faces]).c('b').lw(4).wireframe()

a1 = precision(m1.area(),3) + " \mum\^2"
a2 = precision(m2.area(),3) + " \mum\^2"

vig1 = m1.vignette('Triangle 1\nA=' + a1,
                   point=(2.1,0.7), s=0.012, offset=(-0.3,0.04))
vig2 = m2.vignette('Triangle 2\nA=' + a2,
                   point=(1.9,0.4), s=0.012, offset=(0.2,-0.2))

m3 = m1.clone().wireframe(False).c('tomato').lw(0)

zax = (0,0,1)
v0,v1,v2 = np.insert(np.array(verts2), 2, 0, axis=1)

m3.cutWithPlane(origin=v0, normal=np.cross(zax, v1-v0))
if m3.NPoints():
    m3.cutWithPlane(origin=v1, normal=np.cross(zax, v2-v1))
if m3.NPoints():
    m3.cutWithPlane(origin=v2, normal=np.cross(zax, v0-v2))
vig3 = m3.vignette('Overlap polygon\nA=' + precision(m3.area(),3),
                   point=(2.2,0.6), s=0.012)

show(m1, m2, m3, vig1, vig2, vig3, __doc__,
     axes=1, size=(800,600), zoom=1.3).close()
