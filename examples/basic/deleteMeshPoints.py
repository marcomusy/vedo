"""Remove points and cells from a mesh
which are closest to a specified point.
"""
from vedo import *

pu = load(datadir+'apple.ply')
pu.c('lightgreen').bc('tomato').lw(0.1)

pt = [1, 0.5, 1]
ids = pu.closestPoint(pt, N=200, returnIds=True)

pu.deletePoints(ids, renamePoints=1)

show(Point(pt), pu, __doc__, axes=1)
