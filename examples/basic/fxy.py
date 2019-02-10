'''
Example for function() method.
Draw a surface representing the 3D function specified as a string
or as a reference to an external already existing function.
Red points indicate where the function does not exist.
'''
print(__doc__)
from vtkplotter import Plotter, fxy, sin, cos


def my_z(x,y): 
    return sin(2*x*y) * cos(3*y)/2


vp = Plotter(shape=(2,2), sharecam=False, bg='snow')

# draw at renderer nr.0 the first actor, show it with a texture
# an existing function z(x,y) can be passed:
f1 = fxy(my_z)
vp.show(f1, at=0)

# c=None shows the original z-scalar color scale. No z-level lines.
f2 = fxy(lambda x,y: sin(x*y), c=None, zlevels=None, texture=None, wire=1)
vp.show(f2, at=1)

# red dots are shown where the function does not exist (y>x):
# if vp is set to verbose, sympy calculates derivatives and prints them:
f3 = fxy('sin(3*x)*log(x-y)/3')
vp.show(f3, at=2)

# specify x and y ranges and z vertical limits:
f4 = fxy('log(x**2+y**2 - 1)', x=[-2,2], y=[-2,2], zlimits=[-1,1.5])
vp.show(f4, at=3, axes=2, interactive=1)


