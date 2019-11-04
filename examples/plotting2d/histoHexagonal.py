from vtkplotter import *
import numpy as np

N = 2000
x = np.random.randn(N) * 1.0
y = np.random.randn(N) * 1.5

# hexagonal histogram:
histo = hexHistogram(x, y, bins=10,
                     xtitle="x gaussian, s=1.0",
                     ytitle="y gaussian, s=1.5",
                     ztitle="dN/dx/dy",
                     fill=True, cmap='terrain')

# scatter plot, place it at z=7:
pts = Points([x, y], c="black", alpha=0.1).z(7)

# add a formula:
f = r'f(x, y)=A \exp \left(-\left(\frac{\left(x-x_{o}\right)^{2}}'
f+= r'{2 \sigma_{x}^{2}}+\frac{\left(y-y_{o}\right)^{2}}'
f+= r'{2 \sigma_{y}^{2}}\right)\right)'
formula = Latex(f, c='k', s=1.5).rotateZ(90).rotateX(90).pos(1.2,-1,1)

show(histo, pts, formula, axes=1, bg="white", viewup='z')
