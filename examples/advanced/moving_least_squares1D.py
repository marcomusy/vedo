"""1D Moving Least Squares (MLS)
to project a cloud of unordered points
to become a smooth line"""
from vedo import *
import numpy as np

N = 4  # nr. of iterations

# build some initial cloud of noisy points along a line
pts = [ (sin(6*x), cos(2*x)*x, cos(9*x)) for x in np.arange(0,2, .001)]
# pts = [ (0, sin(x), cos(x)) for x in arange(0,6, .002) ]
# pts = [(sqrt(x), sin(x), x/5) for x in arange(0, 16, 0.01)]

pts += np.random.randn(len(pts), 3) /20  # add noise
np.random.shuffle(pts)  # make sure points are not ordered

pts = Points(pts, r=5)

plt = Plotter(N=N, axes=1)
plt.at(0).show(pts, __doc__)

for i in range(1, N):
    pts = pts.clone().smoothMLS1D(f=0.4).color(i)

    if i == N-1:
        # at the last iteration make sure points
        # are separated by tol (in % of bbox)
        pts.subsample(0.02)

    plt.at(i).show(pts, f"Iteration {i}, #points: {pts.N()}")

plt.interactive().close()
