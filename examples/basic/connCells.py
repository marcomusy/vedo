"""
Parse mesh with connectedCells()
"""
from vtkplotter import *

N = 12
sub = 0
tol = 0.02

actor = load(datadir+'250.vtk')
show(
     actor.wire().c('blue'),
     actor.boundaries(),
     Point(actor.getPoint(30)),
     actor.connectedCells(30), # cells polygons at vertex nr.30
     )

s = actor.subdivide(sub).clean(tol)

coords = s.coordinates() 
pactor = Points(coords)

tomerge = []
for p in coords:
    ipts = s.closestPoint(p, N=N, returnIds=True)
    pts = coords[ipts]
    
    d = delaunay2D(pts, mode='fit').c('blue').wire()
    
    piece = d.connectedCells(0, returnIds=False)

    show(pactor, d, piece, Point(p, c='r'), interactive=0)

    tomerge.append(piece)

show(mergeActors(tomerge).clean(), interactive=1)

