## Example
# make a textured floor, a lamp post, and load a mesh of a car
# make copies of the car, rotate and move them in a loop
# vp.render() is used inside the loop, itadds the actor to list vp.actors,
# rate=10 limits the speed of the loop to maximum 10 fps

from __future__ import division, print_function
import vtkplotter

vp = vtkplotter.Plotter(verbose=0, axes=0)

vp.plane(pos=(4,0,-.45), sx=12, texture='metalfloor1')

# load and set its position (methods can be concatenated)
vp.load('data/shapes/lamp.vtk').pos([1.7, -0.4, 2])

a = vp.load('data/shapes/porsche.ply', c='r').rotateX(90) 
a.normalize() # set actor at origin and scale size to 1

print ('Scene is ready, press q to continue')
vp.show()

for i in range(1, 10):
    b = a.clone(c='aqua', alpha=.04*i)
    b.rotateX(-20*i).rotateY(-10*i).pos([i, i/2, i/2])
    vp.render(b, rate=10)  # add actor b, maximum frame rate in hertz
    print (i, 'time:', vp.clock, 's')
vp.show()
