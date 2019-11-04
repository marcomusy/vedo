"""Markers set, analogous to matplotlib"""
from vtkplotter import *

symbols = ['.', 'p','*','h','D','d','o','v','^','>','<','s','x','+','a']

for i,s in enumerate(symbols):
    Marker(s, filled=True).x(i*0.6).color(i)
    Text(s, pos=[i*0.6,-0.6,0], s=0.12, depth=0).color('k')
Text(__doc__)

show(..., bg='w', size=(1500,300), axes=0, zoom=5)
