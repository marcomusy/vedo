'''
Add an array of cell and point IDs.
'''
print(__doc__)

from vtkplotter import *

act = Spring().addIDs()

print(act.scalars(),'press k to apply point scalars.')

show(act, viewup='z', verbose=0)
