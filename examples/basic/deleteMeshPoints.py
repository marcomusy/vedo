"""
Remove the 200 points (and cells) from a mesh
which are closest to a specified point.
"""
from vtkplotter import *

pu = load(datadir+'pumpkin.vtk')
pu.c('lightgreen').bc('tomato').lw(.1).lc('gray')

pt = [1, 0.5, 1]
ids = pu.closestPoint(pt, N=200, returnIds=True)

pu.deletePoints(ids)

show(Point(pt), pu, Text(__doc__), bg='white')
