"""Subscripts and superscripts in 3D Text"""
from vtkplotter import *

Text("H_2 O at 95 ^o C", c='db', useSubScripts=True)
Text("Avogadro's nr = 6.022e+23 (italics)", italic=1, c='k').y(-3)
Text("Grav. acceleration = 98.0e-01 m/s^2", c='dr').y(-6)
Text("Gaussian kernel is e^-(x-x0)^2/2s", c='dg', depth=0.2).y(-9)
Text2D(__doc__, bg='r')

show(..., azimuth=30, bg2='lb', axes=7, size=(1000,500), zoom=2)
