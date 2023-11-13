"""Identify and fill holes of an input mesh.
Holes are identified by locating boundary edges, linking them
together into loops, and then triangulating the resulting loops."""
from vedo import Mesh, show, dataurl
a = Mesh(dataurl+"bunny.obj").lw(1).bc('red')

b = a.clone() # make a copy
b.fill_holes(size=0.1).color("lb").bc('red5')

show(a, b, __doc__, elevation=-40).close()
