"""Project a spherical-like object onto a plane"""
from vtkplotter import *

e = Ellipsoid()

ef = projectSphereFilter(e).normalize().wireframe(True)

show(e, ef, __doc__)
