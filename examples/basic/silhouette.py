"""Show the silhouette of a mesh
as seen along a specified direction.
"""
from vtkplotter import *

s = Hyperboloid().rotateX(20)

sx = s.clone().projectOnPlane('yz').addPos(-2,0,0).c('r')
sy = s.clone().projectOnPlane('zx').addPos(0,-2,0).c('g')
sz = s.clone().projectOnPlane('xy').addPos(0,0,-2).c('b')

show(s, sx, sy, sz,
     sx.silhouette(),
     sy.silhouette(),
     sz.silhouette(),
     Text(__doc__), axes=1, viewup='z', bg='w')