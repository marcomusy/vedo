"""Generate the silhouette of a mesh
as seen along a specified direction
"""
from vtkplotter import *

s = Hyperboloid().rotateX(20)

sx = s.clone().projectOnPlane('x').c('r').x(-3)
sy = s.clone().projectOnPlane('y').c('g').y(-3)
sz = s.clone().projectOnPlane('z').c('b').z(-3)

show(s,
     sx, sx.silhouette(),
     sy, sy.silhouette(),
     sz, sz.silhouette(),
     Text2D(__doc__),
     axes={'zxGrid':True},
     viewup='z')
