"""
Mesh smoothing with two different methods.
"""
print(__doc__)
from vedo import Plotter, datadir

vp = Plotter(shape=(1, 3))

# Load a mesh and show it
m0 = vp.load(datadir+"embryo.tif", threshold=True, c="v").normalize().lw(0.1)
vp.show(m0, at=0)

# Adjust mesh using Laplacian smoothing
m1 = m0.clone().smoothLaplacian(niter=20, relaxfact=0.1, edgeAngle=15, featureAngle=60)
m1.color("crimson").legend("laplacian")
vp.show(m1, at=1)

# Adjust mesh using a windowed sinc function interpolation kernel
m2 = m0.clone().smoothWSinc(niter=15, passBand=0.1, edgeAngle=15, featureAngle=60)
m2.color("seagreen").legend("window sinc")
vp.show(m2, at=2)

vp.renderers[0].SetBackground(0.8, 1, 1)  # set first renderer color
vp.show(zoom=1.4, interactive=True)
