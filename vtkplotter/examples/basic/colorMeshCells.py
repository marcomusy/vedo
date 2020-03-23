"""cellColors(mode='colors')
Colorize faces of a mesh one by one,
passing a 1-to-1 list of colors and
optionally a list of transparencies.
"""
from vtkplotter import *

################################################
sc = Torus(res=9).lw(0.1)

cols, alphas = [], []
for i in range(sc.NCells()):
    cols.append(i) # i-th color
    alphas.append(i/sc.NCells())

sc.cellColors(cols, alpha=alphas, mode='colors')
print('all mesh arrays:', sc.getArrayNames())

tc = Text2D(__doc__, c='k')

show(sc, tc, at=0, N=2)

################################################
sv = Torus(res=5).lw(0.1)

cols = [i for i in range(sv.NPoints())]

sv.pointColors(cols, mode='colors')
print('all mesh arrays:', sv.getArrayNames())

tv = Text2D('''pointColors(mode='colors')
	tries to interpolate inbetween vertices''', c='k')

show(sv, tv, at=1, interactive=True)
