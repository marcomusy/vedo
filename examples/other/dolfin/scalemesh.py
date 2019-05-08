"""
Scale a mesh asymmetrically in one coordinate
"""
from dolfin import *
from mshr import *

domain = Rectangle(Point(0.0, 0.0), Point(5.0, 0.01))
mesh = generate_mesh(domain, 20)
V = FunctionSpace(mesh, "CG", 2)

e = Expression("sin(2*pi*(x[0]*x[0]+x[1]*x[1]))", degree=2)
f = interpolate(e, V)

####################################################
from vtkplotter.dolfin import plot

plt = plot(f,
           warpZfactor=0.05, # add z elevation proportional to expression
           style=1,
           lw=0,
           xtitle = 'y-coord is scaled by factor 100',
           text=__doc__,
           interactive=False)
plt.actors[0].scale([1,100,1]) # retrieve actor object and scale y
plt.show(interactive=True) # refresh scene