"""Align bounding boxes"""
from vedo import *

eli = Ellipsoid().alpha(0.4)
cube= Cube().pos(3,2,1).rotateX(10).rotateZ(10).wireframe()

eli.alignToBoundingBox(cube, rigid=0)

axes1 = Axes(eli, c='db', htitle='ellipsoid box')
axes2 = Axes(cube, c='dg', htitle='cube box')

show(eli, cube, axes1, axes2, __doc__).close()