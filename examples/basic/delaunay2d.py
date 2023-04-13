"""Delaunay 2D meshing
with point loops defining holes"""
from vedo import *

# Generate a grid and add gaussian noise to it
# then extract the points from the grid and store them in the variable gp
gp = Grid().add_gaussian_noise([0.5,0.5,0]).points()

# Define two internal holes using point ids
ids = [[24,35,36,37,26,15,14,25], [84,95,96,85]]

# Create a point cloud object using the extracted points gp
pts = Points(gp, r=6).c('blue3')

# Use the Delaunay triangulation algorithm to create a 2D mesh 
# from the points in gp, with the given boundary ids
dly = delaunay2d(gp, mode='xy', boundaries=ids).c('w').lc('o').lw(1)

# Create labels for the point ids and set their z to 0.01
labels = pts.labels('id').z(0.01)

show(pts, labels, dly, __doc__, bg="Mint").close()
