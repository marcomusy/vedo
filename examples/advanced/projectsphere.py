"""
Project a spherical-like object onto a plane.
"""
from vtkplotter import *

e = Ellipsoid()

ef = projectSphereFilter(e).normalize().wire(True)

show([e,ef, Text(__doc__)])