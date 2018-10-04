# Mesh smoothing with two different VTK methods
#
# See also analogous Plotter method smoothMLS2D()
# in exammples/advanced/moving_least_squares2D.py 

from vtkplotter import Plotter
from vtkplotter.analysis import smoothLaplacian, smoothWSinc

vp = Plotter(shape=(1,3), axes=4)

# Load a mesh and show it
a0 = vp.load('data/embryo.tif', c='v')
vp.show(a0, at=0)

# Adjust mesh using Laplacian smoothing
a1 = smoothLaplacian(a0).color('crimson').alpha(1)
vp.show(a1, at=1, legend='smoothFilter')

# Adjust mesh using a windowed sinc function interpolation kernel
a2 = smoothWSinc(a0).color('seagreen').alpha(1)
vp.show(a2, at=2, legend='smoothWSinc')

vp.renderers[0].SetBackground(0.8,1,1) # set first renderer color
vp.show(zoom=1.4, interactive=True)
