'''
Add an array of cell and point IDs.
'''
print(__doc__)

from vtkplotter import *

act = helix()

iact = addIDs(act)

print(iact.scalars(),'press k to apply point scalars.')

show(iact, viewup='z', verbose=0)
