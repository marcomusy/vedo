from __future__ import division, print_function
from random import uniform as u, gauss, seed
import plotter
#########################################################
# Example usage of align() method:
# generate two random sets of points as 2 actors 
# and align them using vtkIterativeClosestPointTransform.
# Retrieve the vtk transformation matrix.

vp = plotter.vtkPlotter(shape=[1,2], verbose=0)

N1 = 15  # number of points of first set
N2 = 10  # number of points of second set
x = 1.   # add some randomness

pts1 = [ (u(0,x), u(0,x), u(0,x)+i) for i in range(N1) ]
pts2 = [ (u(0,x)+3, u(0,x)+i/2+2, u(0,x)+i+1) for i in range(N2) ]

act1 = vp.points(pts1, c='b', tags='ids') # ids adds a label to each point
act2 = vp.points(pts2, c='r', tags='ids')

vp.show(at=0, interactive=0)

# find best alignment between the 2 sets of points
alpts1 = vp.align(act1,act2).coordinates()

for i in range(N1): #draw arrows to see where points end up
    vp.arrow(pts1[i], alpts1[i], c='k', s=0.01, alpha=.1) 

print ('transformation matrix:', vp.result['transform'].GetMatrix())

vp.show(at=1, interactive=1)


