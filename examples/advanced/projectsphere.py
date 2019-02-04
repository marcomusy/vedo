"""
Project a spherical-like object onto a plane.
"""
from vtkplotter import *

e = ellipsoid()

ef = projectSphereFilter(e).normalize().wire(True)

show([e,ef, text(__doc__)])