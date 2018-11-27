# histogram2D example 
#
from vtkplotter import Plotter
import numpy as np

vp = Plotter(axes=1, verbose=0)
vp.xtitle = 'x gaussian, s=1.5'
vp.ytitle = 'y gaussian, s=1.0'
vp.ztitle = 'dN/dx/dy'

N = 20000
x = np.random.randn(N)*1.5
y = np.random.randn(N)*1.0

vp.histogram2D(x, y, c='dr', bins=15, fill=False)

pts = list(zip(x, y, np.zeros(N)))
vp.points(pts, c='black', alpha=0.1)

vp.show()
