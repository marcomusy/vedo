"""
This example shows how to use a variant of a 1 dimensional
Moving Least Squares (MLS) algorithm to project a cloud
of unordered points to become a smooth line.
The parameter f controls the size of the local regression.
"""
print(__doc__)
from vedo import *
import numpy as np

N = 6  # nr. of iterations

# build some initial cloud of noisy points along a line
pts = [ (sin(6*x), sin(2*x)/(x+1), cos(9*x)) for x in arange(0,1, .001)]
#pts = [ (0, sin(x), cos(x)) for x in arange(0,6, .002) ]
#pts = [(sqrt(x), sin(x), x/10) for x in arange(0, 16, 0.01)]

pts += np.random.randn(len(pts), 3) /20  # add noise
np.random.shuffle(pts)  # make sure points are not ordered

a = Points(pts).legend("cloud")

show(a, at=0, N=N, axes=5)

for i in range(1, N):
    a = a.clone().smoothMLS1D(f=0.2).color(i).legend("iter #" + str(i))

    # at last iteration make sure points are separated by tol (in % of bbox)
    if i == N-1:
        a.clean(tol=0.01)

    print("iteration", i, "#points:", len(a.points()))
    show(a, at=i)

interactive()
