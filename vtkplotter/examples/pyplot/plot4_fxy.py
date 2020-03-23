from vtkplotter import *

########################################################### REAL
#Draw a surface representing a 2-var function specified
#as a string or as a reference to an external existing function.
#Red points indicate where the function does not exist.

# an existing function z(x,y) can be passed:
def my_z(x, y): return sin(2 * x * y) * cos(3 * y) / 2
f1 = plot(my_z)

# red dots are shown where the function does not exist (y>x):
f2 = plot("sin(3*x)*log(x-y)/3")

# specify x and y ranges and z vertical limits:
f3 = plot("log(x**2+y**2-1)", xlim=[-2,2], ylim=[-1,8], zlim=[-1,None])

show(f1, f2, f3, N=3, axes=1, sharecam=False)


########################################################## COMPLEX
comment = Text2D("""
Vertical axis shows the real part of complex z:
    z = sin(log(x*y))
Color map the value of the imaginary part
(green=positive, purple=negative)"""
)

plt = plot(lambda x,y: sin(log(x*y))/25, mode='complex')

show(plt, comment, viewup='z', newPlotter=True)
