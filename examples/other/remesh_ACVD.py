# Credits:
# https://github.com/akaszynski/pyacvd
# Needs PyACVD:
# pip install pyacvd
#
from vedo import *
from pyvista import wrap
from pyacvd import Clustering

mesh = Sphere(res=50).subdivide().lw(0.2).cutWithPlane().clean()

clus = Clustering(wrap(mesh.polydata()))
clus.cluster(1000, maxiter=100, iso_try=10, debug=False)

pvremesh = clus.create_mesh()

remesh = Mesh(pvremesh).computeNormals()
remesh.color('o6').backColor('v').lw(0.2).shift(1,0,0)

show(mesh, remesh)

#remesh.write('sphere.vtk')
