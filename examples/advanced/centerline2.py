"""Find the center line of a tube with bifurcation"""
from vedo import *

# parameters to play with:
f     = 0.25  # moving least squares fraction of neighbours
niter = 24    # moving least squares iterations
tol   = 0.03  # minimum point distance as fraction

# create a bifurcating mesh
rng = arange(0,1,0.02)
l0  = [vector(0,0,0)*(1-s)+vector(  0,0,1)*s for s in rng]
l0 += [vector(0,0,1)*(1-s)+vector(-.5,0,2)*s for s in rng]
l1  = [vector(0,0,1)*(1-s)+vector(0.5,0,2)*s for s in rng]

t0 = Tube(l0, r=0.1, res=24)
t1 = Tube(l1, r=0.1, res=24)

tube = t0.boolean('+',t1)
tube.computeNormals().alpha(0.2).lineWidth(0.1)

t = tube.clone().clean(tol/2)
for i in range(niter):
    t = t.clone().smoothMLS1D(f=f).c('white').pointSize(10)
    show(t, at=i, N=niter, elevation=-1, bg='bb')
t.clean(tol)

show(tube, t, __doc__, axes=1, bg='bb', new=True)