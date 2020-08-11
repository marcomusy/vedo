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
from vedo.dolfin import plot

plt = plot(f,
           xtitle='y-coord is scaled by factor 100',
           scaleMeshFactors=(0.01, 1, 1),
           style=1,
           lw=0,
           warpZfactor=0.001,
           scalarbar='horizontal',
           axes={'xTitleOffset':0.2},
           text=__doc__,
           )
