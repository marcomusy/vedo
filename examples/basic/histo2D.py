"""
2D histogram with hexagonal binning.
"""
from vtkplotter import *
import numpy as np

vp = Plotter(axes=1, verbose=0, bg="w")
vp.xtitle = "x gaussian, s=1.0"
vp.ytitle = "y gaussian, s=1.5"
vp.ztitle = "dN/dx/dy"

N = 20000
x = np.random.randn(N) * 1.0
y = np.random.randn(N) * 1.5

histo = histogram2D(x, y, bins=10, fill=True)

pts = Points([x, y, np.zeros(N)+6], c="black", alpha=0.01)

f = r'f(x, y)=A \exp \left(-\left(\frac{\left(x-x_{o}\right)^{2}}'
f+= r'{2 \sigma_{x}^{2}}+\frac{\left(y-y_{o}\right)^{2}}'
f+= r'{2 \sigma_{y}^{2}}\right)\right)'

formula = Latex(f, c='k', s=1.5).rotateZ(90).rotateX(90).pos(1,-1,1)

vp.show(histo, pts, formula, Text(__doc__), viewup="z")
