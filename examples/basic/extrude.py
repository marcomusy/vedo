"""Extruding a polygon along the z-axis"""
from vedo import *

pol = Star().color('y').pos(0,0,0).rotateX(10)

epol = pol.extrude(zshift=1, rotation=10, dR=-0.2, cap=False, res=1)
epol.flat().bc('violet')

show(epol, __doc__, axes=1, viewup='z').close()
