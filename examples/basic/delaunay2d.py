"""Delaunay 2D meshing
with point loops defining holes"""
from vedo import *

gp = Grid().addGaussNoise([0.5,0.5,0]).points()

# Use point ids to define two internal holes
ids = [[24,35,36,37,26,15,14,25], [84,95,96,85]]

pts = Points(gp, r=6).c('blue3')
dly = delaunay2D(gp, mode='xy', boundaries=ids).c('w').lc('o').lw(1)
labels = pts.labels('id').z(0.01)

show(pts, labels, dly, __doc__, bg="Mint").close()
