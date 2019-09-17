"""
Draw a surface representing a 2-var function specified
as a string or as a reference to an external existing function.
Red points indicate where the function does not exist.
"""
print(__doc__)
from vtkplotter import fxy, sin, cos, show


def my_z(x, y):
    return sin(2 * x * y) * cos(3 * y) / 2

# an existing function z(x,y) can be passed:
f1 = fxy(my_z)

# red dots are shown where the function does not exist (y>x):
f2 = fxy("sin(3*x)*log(x-y)/3")

# specify x and y ranges and z vertical limits:
f3 = fxy("log(x**2+y**2-1)", x=[-2,2], y=[-1,8], zlimits=[-1,None])

show(f1, f2, f3, N=3, axes=1, sharecam=False, bg="w")
