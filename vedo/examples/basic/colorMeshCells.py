"""Colorize faces of a Mesh
passing a 1-to-1 list of colors and
optionally a list of transparencies"""
from vedo import *

tor = Torus(res=9).lineWidth(0.1)

cols, alphas = [], []
for i in range(tor.NCells()):
    cols.append(i)                # i-th color
    alphas.append(i/tor.NCells()) # a transparency value

tor.cellIndividualColors(cols, alphas)
print('Mesh arrays:', tor.getArrayNames())

show(tor, __doc__)
