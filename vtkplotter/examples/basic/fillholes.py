"""Identifies and fills holes in input mesh.
Holes are identified by locating boundary edges, linking them
together into loops, and then triangulating the resulting loops.
size: approximate limit to the size of the hole that can be filled.
"""
from vtkplotter import load, show, Text2D, datadir

a = load(datadir+"bunny.obj").lw(0.1)

b = a.clone().pos(.2,0,0).fillHoles(size=0.1)
b.color("lb").legend("filled mesh")

show(a, b, Text2D(__doc__), elevation=-70)
