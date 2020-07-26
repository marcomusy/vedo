"""Extract all image voxels as points"""
from vedo import *

v = load(datadir+'vase.vti') # Volume

pts = v.toPoints().printInfo() # returns a Mesh(vtkActor)

scalars = pts.getPointArray(0)
pts.cmap('afmhot_r', scalars)

show([(v,__doc__), pts], N=2, viewup='z', zoom=1.5)