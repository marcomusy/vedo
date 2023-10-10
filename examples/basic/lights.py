"""Set custom lights to a 3D scene"""
from vedo import *

man = Mesh(dataurl + 'man.vtk')
man.c('white').lighting('glossy')

p1 = Point([1,0,1], c='y')
p2 = Point([0,0,2], c='r')
p3 = Point([-1,-0.5,-1], c='b')
p4 = Point([0,1,0], c='k')

# Add light sources at the given positions
l1 = Light(p1, c='y') # p1 can simply be [1,0,1]
l2 = Light(p2, c='r')
l3 = Light(p3, c='b')
l4 = Light(p4, c='w', intensity=0.5)

show(
    man, l1, l2, l3, l4, p1, p2, p3, p4,
    __doc__, axes=1, viewup='z',
).close()
