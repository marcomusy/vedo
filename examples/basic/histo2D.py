"""
2D histogram with hexagonal binning.
"""
print(__doc__)
from vtkplotter import Plotter, histogram2D, Points, Latex
import numpy as np

vp = Plotter(axes=1, verbose=0, bg="w")
vp.xtitle = "x gaussian, s=1.0"
vp.ytitle = "y gaussian, s=1.5"
vp.ztitle = "dN/dx/dy"

N = 20000
x = np.random.randn(N) * 1.0
y = np.random.randn(N) * 1.5

histo = histogram2D(x, y, c="dr", bins=15, fill=False)

pts = Points([x, y, np.zeros(N)], c="black", alpha=0.1)

f = r'f(x, y)=A \exp \left(-\left(\frac{\left(x-x_{o}\right)^{2}}{2 \sigma_{x}^{2}}+\frac{\left(y-y_{o}\right)^{2}}{2 \sigma_{y}^{2}}\right)\right)'

formula = Latex(f, c='k', s=2).rotateZ(90).pos(5,-2,1)

vp.show(histo, pts, formula, viewup="z")
