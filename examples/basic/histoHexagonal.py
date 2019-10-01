"""2D histogram with hexagonal binning."""
from vtkplotter import *
import numpy as np

N = 2000
x = np.random.randn(N) * 1.0
y = np.random.randn(N) * 1.5

# hexagonal histogram
histo = hexHistogram(x, y, bins=10, fill=True, cmap='terrain')

# scatter plot:
pts = Points([x, y, np.zeros(N)+6], c="black", alpha=0.05)

f = r'f(x, y)=A \exp \left(-\left(\frac{\left(x-x_{o}\right)^{2}}'
f+= r'{2 \sigma_{x}^{2}}+\frac{\left(y-y_{o}\right)^{2}}'
f+= r'{2 \sigma_{y}^{2}}\right)\right)'
formula = Latex(f, c='k', s=1.5).rotateZ(90).rotateX(90).pos(1,-1,1)

#settings.useParallelProjection = True
settings.xtitle = "x gaussian, s=1.0"
settings.ytitle = "y gaussian, s=1.5"
settings.ztitle = "dN/dx/dy"

show(histo, pts, formula, Text(__doc__),
     axes=1, verbose=0, bg="white", viewup='z')
