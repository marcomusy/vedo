"""
Extruding a polygon
along the z-axis
"""
from vtkplotter import *

pol = Star().pos(0,0,0).rotateX(10)

epol = extrude(pol, zshift=1, rotation=10, dR=-0.2, cap=False, res=1)
epol.bc('violet')

show(epol, Text(__doc__), axes=1, bg='white', viewup='z')
