'''Draw a z = f(x,y) surface specified as
a string or as a reference to an external function.
Red points indicate where the function does not exist!'''
from vedo import dataurl, sin, cos, log, show, Text2D
from vedo.pyplot import plot

doc = Text2D(__doc__, pos='bottom-left', c='darkgreen', font='Quikhand')

############################################################### REAL
def f(x, y):
    return sin(2*x*y) * cos(3*y)/2
f1 = plot(f, c='summer') # use a colormap

# red dots are shown where the function does not exist (y>x):
def f(x, y):
    return sin(3*x) * log(x-y)/3
f2 = plot(f, texture=dataurl+'textures/paper3.jpg')

# specify x and y ranges and z vertical limits:
def f(x, y):
    return log(x**2+y**2-1)
f3 = plot(
    f,
    xlim=[-2,2],
    ylim=[-1,8],
    zlim=[-1,None],
    texture=dataurl+'textures/paper1.jpg',
)

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

show(plt, comment, viewup='z').close()
