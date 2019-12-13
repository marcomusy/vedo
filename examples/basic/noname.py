'''
for i in range(10):
    Cone().x(i) # no variable assigned!
show(...) # show all sofar created objs
'''
from vtkplotter import Cone, Text, show

for i in range(10):
    Cone().x(2*i).color(i) # no variable assigned

Text(__doc__)

# three points, aka ellipsis, retrieves the list of all created objects
show(..., axes=1, viewup='z')
