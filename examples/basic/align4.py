"""Align a set of curves in space
with Procrustes method"""
from vedo import *

splines = load(dataurl+'splines.npy')  # file contains a list of vedo.Lines

procus = procrustesAlignment(splines, rigid=False)
alignedsplines = procus.unpack()  # unpack Assembly into a python list
mean = procus.info['mean']
lmean = Line(mean, lw=4, c='b').z(0.001) # z-shift it to make it visible

for l in alignedsplines:
    darr = mag(l.points()-mean)  # distance array
    l.cmap('hot_r', darr, vmin=0, vmax=0.007)

alignedsplines += [lmean, __doc__]

show([splines, alignedsplines], N=2, sharecam=False, axes=1).close()

