'''
2D histogram with hexagonal binning.
'''
print(__doc__)
from vtkplotter import Plotter, histogram2D, Points
import numpy as np

vp = Plotter(axes=1, verbose=0, bg='w')
vp.xtitle = 'x gaussian, s=1.0'
vp.ytitle = 'y gaussian, s=1.5'
vp.ztitle = 'dN/dx/dy'

N = 20000
x = np.random.randn(N)*1.0
y = np.random.randn(N)*1.5

vp.add(histogram2D(x, y, c='dr', bins=15, fill=False))

vp.add(Points([x, y, np.zeros(N)], c='black', alpha=0.1))

vp.show(viewup='z')
