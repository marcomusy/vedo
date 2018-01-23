## Example
## make a textured floor, a cylinder post, and load a mesh of a car
## make copies of the car, rotate and move them in a loop
## vp.render() adds the actor to the list vp.actors
## rate=10 limits the speed of the loop

from __future__ import division, print_function
import plotter

vp = plotter.vtkPlotter(verbose=0)

vp.plane(pos=(4,0,-.35), s=12, alpha=.9, texture='metalfloor1')
vp.cylinder(pos=(1.6,-.4,1.1), radius=0.1, height=3, texture='wood1')

a = vp.load('data/shapes/porsche.ply', c='r', alpha=1)
a.rotateX(-90).normalize() # set actor at origin and scale size to 1

print ('Scene is ready, press q to continue')
vp.show()

for i in range(1, 10):
    b = a.clone(c='aqua', alpha=.04*i)
    b.rotateX(20*i).rotateY(10*i).pos([i, i/2, i/2])
    vp.render(b, rate=10)  # add actor b, maximum frame rate in hertz
    print (i, 'time:', vp.clock, 's')
vp.show()
