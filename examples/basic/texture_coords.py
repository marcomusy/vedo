"""Assign texture coordinates to a polygon"""
from vedo import *

settings.defaultFont = 'Bongas'

# define a polygon of 4 vertices:
polygon = [
    [(82, 92, 47), (87, 88, 47), # x,y,z of vertices
     (93, 95, 47), (88, 99, 47)],
    [[0, 1, 2, 3]],              # vertex connectivity
]

# texture coordinates, one (u,v) pair for each vertex:
tc = [(0,0), (1,0), (1,1), (0,1)]
#tc = [(0,0), (2,0), (2,2), (0,2)]

# create the Mesh object (a rectangle)
m = Mesh(polygon)

# apply texture to m
fpath = download('https://vedo.embl.es/examples/data/images/dog.jpg')
m.texture(fpath,
          tcoords=tc,
          interpolate=True,
          repeat=True,      # when tcoords extend beyond [0,1]
          edgeClamp=False,  #  only used when repeat is False
)

show(m, __doc__, axes=1).close()