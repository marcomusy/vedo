from vedo import Latex, show
from vedo.pyplot import histogram
import numpy as np

N = 2000
x = np.random.randn(N) * 1.0
y = np.random.randn(N) * 1.5

# hexagonal binned histogram:
histo = histogram(x, y, bins=10, mode="hexbin", fill=True, cmap="terrain")
# add a formula:
f  = r"f(x, y)=A \exp \left(-\left(\frac{\left(x-x_{o}\right)^{2}}"
f += r"{2 \sigma_{x}^{2}}+\frac{\left(y-y_{o}\right)^{2}}"
f += r"{2 \sigma_{y}^{2}}\right)\right)"

formula = Latex(f, c="k", s=1.5)
formula.rotate_x(90).rotate_z(90).pos([-4, -5, 2])

show(histo, formula, axes=1, viewup="z").close()
