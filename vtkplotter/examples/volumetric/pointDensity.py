"""Density field as a Volume from a point cloud"""

from vtkplotter import *

t1 = Text2D("Original mesh")
s  = load(datadir+'spider.ply')

t2 = Text2D("Density field as a Volume from point cloud")
v  = pointDensity(s).mode(1).printInfo()

show([ [s,t1], [v,t2] ], N=2)