'''
Make a textured floor, a lamp post, and load a mesh of a car
make copies of the car, rotate and move them in a loop.
rate=10 limits the speed of the loop to maximum 10 fps
'''
from __future__ import division, print_function
from vtkplotter import Plotter, Plane, Text

vp = Plotter(interactive=0, axes=0)

vp.add(Plane(pos=(4,0,-.45), sx=12, texture='metalfloor1'))

# load and set its position (methods can be concatenated)
vp.load('data/shapes/lamp.vtk').pos([1.7, -0.4, 2])

a = vp.load('data/shapes/porsche.ply', c='r').rotateX(90) 
a.normalize() # set actor at origin and scale size to 1

for i in range(1, 10):
    b = a.clone().color('aqua').alpha(.04*i)
    b.rotateX(-20*i).rotateY(-10*i).pos([i, i/2, i/2])
    vp.add(b)            # add actor b
    vp.show(rate=10)     # maximum frame rate in hertz
    print(i, 'time:', vp.clock, 's')

vp.add(Text(__doc__))

vp.show(interactive=1)

