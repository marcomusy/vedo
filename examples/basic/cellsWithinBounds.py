"""
Find cells within specified bounds in x, y and z.
"""
from vtkplotter import *

mesh = load(datadir+'shark.ply').normalize()
mesh.color('aqua').lineWidth(0.1)

x1, x2 = -0.1,0.3

ids = mesh.findCellsWithin(xbounds=(x1,x2), c='tomato')

printc('IDs of cells within bounds:', ids, c='g')

p1 = Plane(normal=(1,0,0), sx=2, sy=2).x(x1).c('gray').alpha(0.5)
p2 = Plane(normal=(1,0,0), sx=2, sy=2).x(x2).c('gray').alpha(0.5)

show(mesh, p1, p2, Text(__doc__), axes=1)
