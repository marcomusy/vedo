"""Split a mesh by connectivity and
order the pieces by increasing surface area"""
from vedo import dataurl, Volume, show

em = Volume(dataurl+"embryo.tif").isosurface(80)

# return the list of the largest 10 connected meshes:
splitem = em.split(maxdepth=40)[0:9]

show(splitem, __doc__, axes=1, viewup='z').close()
