"""Using nevergrad package to find
the minimum of the 2-variable function:
z = (x-1)**2 + (y-1)**2 + 9*sin(y-1)**2
"""
from vedo import *
from vedo.pyplot import plot
import nevergrad as ng # install with: pip install nevergrad


def f(x,y):
    z = (x-1)**2 + (y-1)**2 + 9*sin(y-1)**2 + 1
    return z/12

def func(v): return f(v[0],v[1])

def callbk(optimizer, v, value):
    global minv
    if value<minv:
        pts.append([v.value[0], v.value[1], value])
        minv = value

optimizer = ng.optimizers.OnePlusOne(parametrization=2, budget=100)

pts, minv = [], 1e30
optimizer.register_callback("tell", callbk)

# define a constraint on first variable of x:
#optimizer.parametrization.register_cheap_constraint(lambda v: v[0]>-3)

res = optimizer.minimize(func)  # best value
printc('Minimum at:', res.value)

ln = Line(pts, lw=3, c='r')
fu = plot(f, xlim=[-3,4], ylim=[-3,4], alpha=0.5)

show(fu, ln, __doc__)
