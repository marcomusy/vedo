#!/usr/bin/env python
#
# Example of use of function() method.
# Draw a surface representing the 3D function specified as a string
# or as a reference to an external already existing function.
# Red points indicate where the function does not exist.
#
import math
import plotter


def my_z(x,y): 
    return math.sin(2*x*y) * math.cos(3*y)/2


vp = plotter.vtkPlotter(shape=(2,2))
vp.commoncam = False

# draw at renderer nr.0 the first actor, show it with a texture
# an existing function z(x,y) can be passed:
f1 = vp.fxy(my_z, texture='paper')
vp.show(f1, at=0, interactive=0)

# c=None shows the original z-scalar color scale. No z-level lines.
# if vp is set to verbose, sympy calculates derivatives and prints them:
f2 = vp.fxy(lambda x,y: math.sin(x*y), c=None, zlevels=None)
vp.show(f2, at=1)

# red dots are shown where the function does not exist (y>x):
f3 = vp.fxy('sin(3*x)*log(x-y)/3')
vp.show(f3, at=2)

# specify x and y ranges and z vertical limits:
f4 = vp.fxy('log(x**2+y**2 - 1)', x=[-2,2], y=[-2,2], zlimits=[-1,1.5])
vp.show(f4, at=3, interactive=1)


