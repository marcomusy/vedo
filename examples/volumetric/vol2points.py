"""Extract all image voxels as points"""
from vtkplotter import *

t = Text(__doc__, c='white')
v = load(datadir+'vase.vti') # Volume

pts = volumeToPoints(v).printInfo() # returns an Actor

scalars = pts.scalars(0)
pts.pointColors(scalars, cmap='jet')

show([(v,t), pts], N=2, viewup='z', zoom=1.5)

