# Credits:
# https://github.com/akaszynski/PyACVD
# Needs PyACVD:
# pip install PyACVD
#
from vtkplotter import *
from PyACVD import Clustering

amesh = Sphere(res=50)

# Create clustering object
poly = amesh.clone().triangle().clean().polydata()
cobj = Clustering.Cluster(poly)

# Generate clusters
cobj.GenClusters(1000, max_iter=10000, subratio=10)
cobj.GenMesh()

remesh = Actor(cobj.ReturnMesh(), c='o', bc='v', computeNormals=True)
remesh.flipNormals()

show(amesh.lw(0.2), remesh.lw(0.2), N=2)

#remesh.write('sphere.vtk')
