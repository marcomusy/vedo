"""Extract all image voxels as points"""
from vedo import *

v = Volume(dataurl+'vase.vti')

pts = v.toPoints().printInfo() # returns Points

scalars = pts.getPointArray(0)
pts.cmap('afmhot_r', scalars)

show([(v,__doc__), pts], N=2, viewup='z', axes=1).close()