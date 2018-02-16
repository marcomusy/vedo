#!/usr/bin/env python2
# 
# Example of boolean operations with actors or polydata
#
import plotter

# declare the instance of the class
vp = plotter.vtkPlotter(shape=(2,2), interactive=0)

# build to sphere actors 
s1 = vp.sphere(pos=[-.7,0,0], c='r', alpha=0.5)
s2 = vp.sphere(pos=[0.7,0,0], c='g', alpha=0.5)

# make 3 different possible operations:
b1 = vp.boolActors(s1, s2, 'intersect', c='m')
b2 = vp.boolActors(s1, s2, 'plus', c='b', wire=True)
b3 = vp.boolActors(s1, s2, 'minus', c=None)

# show the result in 4 different subwindows 0->3
vp.show([s1,s2], at=0, legend='2 spheres')
vp.show(b1, at=1, legend='intersect')
vp.show(b2, at=2, legend='plus')
vp.show(b3, at=3, legend='minus')
vp.addScalarBar() # adds a scalarbar to the last actor 
vp.show(interactive=1)