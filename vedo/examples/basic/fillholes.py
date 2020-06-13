"""Identify and fill holes of an input mesh.
Holes are identified by locating boundary edges, linking them
together into loops, and then triangulating the resulting loops.
"""
from vedo import load, show, datadir

a = load(datadir+"bunny.obj").lw(0.1)

# size = approximate limit to the size of the hole to be filled.
b = a.clone().pos(.2,0,0).fillHoles(size=0.1)
b.color("lb").legend("filled mesh")

show(a, b, __doc__, elevation=-70)
