"""Surface plotting in spherical coordinates

Spherical harmonic function is:
Y(l=2, m=0) = 3\dotcos\^2\theta - 1
(red points are made NaN on purpose)
"""
from vedo import *
from vedo.pyplot import plot
import numpy as np

def rhofunc(theta, phi):
    if theta < 0.2:
        return np.nan # make some points invalid
    #return cos(theta)**2                       # Y(l=1 m=0)
    return (3*cos(theta)**2 - 1)**2             # Y(l=2 m=0)
    #return (5*cos(theta)**3 - 3*cos(theta))**2 # Y(l=3 m=0)

# Build the plot,
#  return an Assembly of 3 meshes, the unit
#  grid sphere, the surface rho(theta, phi) and
#  the red Points where rho is a complex number:
spl = plot(rhofunc, mode='spheric', cmap='viridis')

show(spl, __doc__, axes=12, viewup='z').close()
