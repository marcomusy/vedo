"""Find the overlap area of 2 triangles"""
from vtkplotter import Actor, Text, show
import numpy as np

verts1 = [(1.9, 0.5), (2.1, 0.8), (2.4, 0.4)]
verts2 = [(2.3, 0.8), (1.8, 0.4), (2.1, 0.3)]
faces = [(0,1,2)]

a1 = Actor([verts1, faces]).c('g').lw(4).wireframe()
a2 = Actor([verts2, faces]).c('b').lw(4).wireframe()

a3 = a1.clone().wireframe(False).c('tomato').lw(0)

zax = (0,0,1)
v0,v1,v2 = np.insert(np.array(verts2), 2, 0, axis=1)

a3.cutWithPlane(origin=(v0+v1)/2, normal=np.cross(zax, v1-v0))
if a3.NPoints():
    a3.cutWithPlane(origin=(v1+v2)/2, normal=np.cross(zax, v2-v1))
if a3.NPoints():
    a3.cutWithPlane(origin=(v2+v0)/2, normal=np.cross(zax, v0-v2))

print("Area of overlap:", a3.area())
show(a1, a2, a3, Text(__doc__), bg='w', axes=8, verbose=0)
