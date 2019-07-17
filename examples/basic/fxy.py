"""
Example for function() method.
Draw a surface representing the 3D function specified as a string
or as a reference to an external already existing function.
Red points indicate where the function does not exist.
"""
print(__doc__)
from vtkplotter import Plotter, fxy, sin, cos, show


def my_z(x, y):
    return sin(2 * x * y) * cos(3 * y) / 2


# draw at renderer nr.0 the first actor, show it with a texture
# an existing function z(x,y) can be passed:
f1 = fxy(my_z)

# red dots are shown where the function does not exist (y>x):
# if vp is set to verbose, sympy calculates derivatives and prints them:
f2 = fxy("sin(3*x)*log(x-y)/3")

# specify x and y ranges and z vertical limits:
f3 = fxy("log(x**2+y**2 - 1)", x=[-2, 2], y=[-2, 2], zlimits=[-1, 1.5])

show(f1, f2, f3, N=3, axes=1, sharecam=False, bg="w")
