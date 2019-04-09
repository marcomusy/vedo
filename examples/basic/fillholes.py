"""
Identifies and fills holes in input mesh.
Holes are identified by locating boundary edges, linking them
together into loops, and then triangulating the resulting loops.
size: approximate limit to the size of the hole that can be filled.
"""
from vtkplotter import load, show, Text, datadir

a = load(datadir+"shapes/bunny.obj")

b = a.clone().fillHoles(size=0.1)
b.color("b").wire(True).legend("filled mesh")

doc = Text(__doc__)

show(a, b, doc, elevation=-70)
