from __future__ import division, print_function
import plotter


vp = plotter.vtkPlotter(verbose=0)
vp.plane(pos=(4,0,-.35), s=12, alpha=.9, texture='metalfloor1')
vp.cylinder(pos=(1.6,-.4,1.1), radius=0.1, height=3, texture='wood1')
a = vp.load('data/shapes/porsche.ply', c='r', alpha=1) 
a.rotateX(-90)
a.normalize() # put actor at origin and scale ave size to 1
print ('Scene is ready, press q to continue')
vp.show()

for i in range(1, 10):
    b = a.clone(c='aqua', alpha=.02*i)
    b.normalize()
    b.addpos([i, i/2, i/4]) # move in x
    b.rotateX(-90+ i*18 )   # angle in degrees
    b.rotateY( i*10 ) 
    vp.addActor(b)
    vp.render(rate=10)      # maximum frame rate in hertz
    print (i, 'time:', vp.clock, 's')
vp.show()
