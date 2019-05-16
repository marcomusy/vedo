"""
Mesh smoothing with two different VTK methods.

See also analogous Plotter method smoothMLS2D()
in exammples/advanced/moving_least_squares2D.py
"""
print(__doc__)
from vtkplotter import Plotter, datadir

vp = Plotter(shape=(1, 3), axes=4)

# Load a mesh and show it
a0 = vp.load(datadir+"embryo.tif", threshold=True, c="v")
vp.show(a0, at=0)

# Adjust mesh using Laplacian smoothing
a1 = a0.clone().smoothLaplacian().color("crimson").alpha(1).legend("laplacian")
vp.show(a1, at=1)

# Adjust mesh using a windowed sinc function interpolation kernel
a2 = a0.clone().smoothWSinc().color("seagreen").alpha(1).legend("window sinc")
vp.show(a2, at=2)

vp.renderers[0].SetBackground(0.8, 1, 1)  # set first renderer color
vp.show(zoom=1.4, interactive=True)
