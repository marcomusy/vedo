"""Split a mesh by connectivity and
order the pieces by increasing area.
"""
from vedo import *

em = load(datadir+"embryo.tif").isosurface(80)

# return the list of the largest 10 connected meshes:
splitem = em.splitByConnectivity(maxdepth=40)[0:9]

show( [(em, __doc__), splitem], N=2, axes=1 )
