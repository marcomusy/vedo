"""Use iminuit to find the minimum of a 3D scalar field"""
from vedo import show, Point, Line, printc
from iminuit import Minuit
# pip install iminuit  # https://github.com/scikit-hep/iminuit
import numpy as np


def fcn(x, y, z):
    f =  (x - 4) ** 4 + (y - 3) ** 4 + (z - 2) ** 2
    if not vals or f < vals[-1]:
        path.append([x,y,z])
        vals.append(f)
    return f


paths = []
for x,y,z in  np.random.rand(200, 3)*3:
    path, vals = [], []
    m = Minuit(fcn, x=x, y=y, z=z)
    m.errordef = m.LEAST_SQUARES

    # m.simplex()  # run simplex optimiser
    m.migrad()     # run migrad optimiser

    line = Line(path).cmap('jet_r', vals).lw(2).alpha(0.25)
    paths.append(line)

printc('Last optimization output:', c='green7', invert=1)
printc(m, c='green7', italic=1)
show(paths, Point([4,3,2]), __doc__, axes=1).close()

