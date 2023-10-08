"""Delaunay 2D meshing with point loops defining holes"""
from vedo import *

# Generate a grid and add gaussian noise to it
# then extract the points from the grid and store them in the variable gp
gp = Grid().add_gaussian_noise([0.5,0.5,0]).point_size(8)

# Define two internal holes using point ids
ids = [[24,35,36,37,26,15,14,25], [84,95,96,85]]

# Use the Delaunay triangulation algorithm to create a 2D mesh 
# from the points in gp, with the given boundary ids
dly = gp.generate_delaunay2d(mode='xy', boundaries=ids)
dly.c('white').lc('orange').lw(1)

# Create labels for the point ids and set their z to 0.01
labels = gp.labels('id').z(0.01)

show(gp, labels, dly, __doc__, bg="Mint", zoom='tight').close()
