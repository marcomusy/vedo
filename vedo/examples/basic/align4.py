"""Align a set of curves in space
with Procrustes method
"""
import numpy as np
from vedo import *

splines = load(datadir+'splines.npy') # returns a list of Lines

procus = procrustesAlignment(splines, rigid=False)
alignedsplines = procus.unpack() # unpack Assembly into a python list
mean = procus.info['mean']
lmean = Line(mean, lw=4, c='b')

for l in alignedsplines:
    darr = np.linalg.norm(l.points()-mean, axis=1)
    l.cmap('hot_r', darr, vmin=0, vmax=0.007)

alignedsplines.append(lmean.z(0.001)) # shift it to make it visible
alignedsplines.append(__doc__)

show([splines, alignedsplines], N=2, sharecam=False, axes=1)

