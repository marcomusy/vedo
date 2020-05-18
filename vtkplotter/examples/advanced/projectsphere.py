"""Project a spherical-like object onto a plane"""
from vtkplotter import *

e = Ellipsoid().lw(0.1).alpha(0.8)

ef = projectSphereToPlane(e).normalize().lw(0.1)

show(e, ef, __doc__)
