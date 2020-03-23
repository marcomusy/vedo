"""
Write one or more vtk objects to a numpy file
and read them back. Properties are also saved.
"""
from vtkplotter import *

######################################## load some vtk objects
tcs = load(datadir+'timecourse1d/reference_*.vtk')
n = len(tcs)
printc('loaded list of meshes of length =', n, c='yellow')

show(tcs, Text2D(__doc__))


######################################## write them to 1 numpy file
# let's change mesh colors (lines in this case)
# all properties (position, color, opacity etc) are saved too. 
for i in range(n):
    tcs[i].color(i)

write(tcs, 'timecourse1d.npy')


######################################## read them back
tcs2 = load('timecourse1d.npy') # will return a list
show(tcs2, newPlotter=True, pos=(1100,0), verbose=0)


######################################## inspect numpy file
import numpy as np

fdict = np.load('timecourse1d.npy', allow_pickle=True)[0]

printc('type is: ', type(fdict),  c='y', italic=1)
printc('keys are:', fdict.keys(), c='g', italic=1)

