"""Extract all image voxels as points"""
from vtkplotter import *

t = Text2D(__doc__, c='white')
v = load(datadir+'vase.vti') # Volume

pts = v.toPoints().printInfo() # returns a Mesh(vtkActor)

scalars = pts.getPointArray(0)
pts.pointColors(scalars, cmap='jet')

show([(v,t), pts], N=2, viewup='z', zoom=1.5)