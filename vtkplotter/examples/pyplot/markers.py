"""Markers set, analogous to matplotlib"""
from vtkplotter import *

symbols = ['.', 'p','*','h','D','d','o','v','^','>','<','s','x','+','a']

vp = Plotter(size=(1500,300), axes=0)
for i,s in enumerate(symbols):
    vp += Marker(s, filled=True).x(i*0.6).color(i)
    vp += Text(s, pos=[i*0.6,-0.6,0], s=0.12, depth=0).color('k')
vp += Text2D(__doc__)

vp.show(zoom=5, viewup='2d')
