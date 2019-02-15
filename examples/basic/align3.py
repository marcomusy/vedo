'''
Example usage of align() method:
 generate 3 random sets of points
 and align them using vtkProcrustesAlignmentFilter.
'''
from __future__ import division, print_function
from random import uniform as u

from vtkplotter import Plotter, procrustes, Text, Points

vp = Plotter(shape=[1, 2], verbose=0, axes=2, sharecam=0, bg='w')

N = 15  # number of points
x = 1.  # add some randomness

pts1 = [(u(0, x), u(0, x), u(0, x)+i) for i in range(N)]
pts2 = [(u(0, x)+3, u(0, x)+i/2+2, u(0, x)+i+1) for i in range(N)]
pts3 = [(u(0, x)+4, u(0, x)+i/4-3, u(0, x)+i-2) for i in range(N)]

act1 = Points(pts1, c='r').legend('set1')
act2 = Points(pts2, c='g').legend('set2')
act3 = Points(pts3, c='b').legend('set3')

vp.show([act1, act2, act3], at=0)

# find best alignment among the n sets of Points,
# return an Assembly formed by the aligned sets
aligned = procrustes([act1, act2, act3])

# print(aligned.info['transform'])

vp.show([aligned, Text(__doc__)], at=1, interactive=1)


