'''
Identifies and fills holes in input mesh. 
Holes are identified by locating boundary edges, linking them 
together into loops, and then triangulating the resulting loops.
size: approximate limit to the size of the hole that can be filled.
'''
from vtkplotter import fillHoles, load, show, text

a = load('data/shapes/bunny.obj')

b = fillHoles(a, size=0.1).color('b').wire(True).legend('filled mesh')

doc = text(__doc__)

show([a,b, doc], elevation=-70)