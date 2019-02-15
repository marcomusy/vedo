'''
Use scipy to interpolate the value of a scalar known on a set of points
on a new set of points where the scalar is not defined.
Two interpolation methods are possible: Radial Basis Function, Nearest Point.
'''
from scipy.interpolate import Rbf, NearestNDInterpolator as Near
import numpy as np
# np.random.seed(0)


# a small set of points for which the scalar is given
x, y, z = np.random.rand(3, 20)

scals = z                            # scalar value is just z component

# build the interpolator
itr = Rbf(x, y, z, scals)              # Radial Basis Function interpolator
# itr = Near(list(zip(x,y,z)), scals) # Nearest-neighbour interpolator

# generate a new set of points
t = np.linspace(0, 7, 100)
xi, yi, zi = [np.sin(t)/10+.5, np.cos(t)/5+.5, (t-1)/5]  # an helix

# interpolate scalar values on the new set
scalsi = itr(xi, yi, zi)


from vtkplotter import Plotter, Points, Text
vp = Plotter(verbose=0, bg='w')
vp.add(Points([x, y, z], r=10, alpha=0.5)).pointColors(scals)
vp.add(Points([xi, yi, zi])).pointColors(scalsi)

vp.add(Text(__doc__, pos=1, c='dr'))
vp.show(viewup='z')
