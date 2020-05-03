"""Extract all image voxels as points"""
from vtkplotter import *

v = load(datadir+'vase.vti') # Volume

pts = v.toPoints().printInfo() # returns a Mesh(vtkActor)

scalars = pts.getPointArray(0)
pts.pointColors(scalars, cmap='jet')

show([(v,__doc__), pts], N=2, viewup='z', zoom=1.5)