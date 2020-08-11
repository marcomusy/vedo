# Credits:
# https://github.com/akaszynski/pyacvd
# Needs PyACVD:
# pip install pyacvd
#
from vedo import *
from pyvista import wrap
from pyacvd import Clustering

mesh = Sphere(res=50).subdivide().lw(0.2).normalize().cutWithPlane()

clus = Clustering(wrap(mesh.polydata()))
clus.cluster(1000, maxiter=100, iso_try=10, debug=False)

pvremesh = clus.create_mesh()

remesh = Mesh(pvremesh).computeNormals()
remesh.color('o').backColor('v').lw(0.2)

show(mesh, remesh, N=2)

#remesh.write('sphere.vtk')
