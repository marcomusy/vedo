"""Find the centerline of an arbitrary tube
"""
from vedo import *

# parameters to play with:
f      = 0.3    # moving least squares fraction of neighbours
niter  = 6      # moving least squares iterations
tol    = 0.03   # minimum point distance as fraction

# built some weird tube for test
ln   = [[sin(x), cos(x), x / 2] for x in arange(0,9,0.1)]
rads = [.3*(cos(6*ir/len(ln)))**2+.1 for ir in range(len(ln))]
tube = Tube(ln, r=rads, res=24, c=None, alpha=0.2)

t = tube
for i in range(niter):
    t = t.clone().smoothMLS1D(f=f).c('white').pointSize(5)
    show(t, at=i, N=niter, bg='bb')

# reduce nr of points by imposing a min distance
t.clean(tol)

show(tube, t, __doc__, axes=1, bg='bb', new=True)
