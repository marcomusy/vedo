"""Mirror a mesh along one of the Cartesian axes.

Hover mouse to identify original and mirrored.
"""
from vedo import *

myted1 = load(datadir+"teddy.vtk").flag('original')

myted2 = myted1.clone().mirror("y")
myted2.pos(0,3,0).c("green").flag('mirrored')

show(myted1, myted2, __doc__, axes=2, viewup="z")
