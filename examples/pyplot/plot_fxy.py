'''Draw a z = f(x,y) surface specified as
a string or as a reference to an external function.
Red points indicate where the function does not exist!'''
from vedo import *
from vedo.pyplot import plot

doc = Text2D(__doc__, pos='bottom-left', c='darkgreen', font='Quikhand')

############################################################### REAL
# an existing function z(x,y) can be passed:
def my_z(x, y): return sin(2*x*y) * cos(3*y)/2
f1 = plot(my_z, c='summer') # use a colormap
# f1 = plot(my_z, c='lightblue', bc='tomato')

# red dots are shown where the function does not exist (y>x):
f2 = plot("sin(3*x)*log(x-y)/3")

# specify x and y ranges and z vertical limits:
f3 = plot("log(x**2+y**2-1)", xlim=[-2,2], ylim=[-1,8], zlim=[-1,None])

show([(f1, 'y = sin(2*x*y) * cos(3*y) /2', doc),
      (f2, 'y = sin(3*x)*log(x-y)/3'),
      (f3, 'y = log(x**2+y**2-1)'),
     ], N=3, sharecam=False,
).close()

############################################################## COMPLEX
comment = """Vertical axis shows the real part of complex z:
    z = sin(log(x\doty))
Color map the value of the imaginary part
(green=positive, purple=negative)"""

plt = plot(lambda x,y: sin(log(x*y))/25, mode='complex')

show(plt, comment, viewup='z', new=True).close()
