"""Extract all image voxels as points"""
from vedo import *

v = Volume(dataurl+'vase.vti')

# Convert every voxel to a point cloud sample.
pts = v.topoints().print() # returns Points

scalars = pts.pointdata[0]
pts.cmap('afmhot_r', scalars).point_size(1)

show([(v,__doc__), pts], N=2, viewup='z', bg2='lightblue', axes=1).close()
