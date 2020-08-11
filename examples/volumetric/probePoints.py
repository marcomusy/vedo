"""Probe a voxel dataset at specified points
and plot a histogram of the values
"""
from vedo import *
from vedo.pyplot import histogram
import numpy as np

vol = load(datadir+'embryo.slc')

pts = np.random.rand(4000, 3)*256

mpts = probePoints(vol, pts).pointSize(3).printInfo()

scals = mpts.getPointArray()

h = histogram(scals, xlim=(5,120), xtitle='probed voxel value')
h.scale(2.2)

show(vol, mpts, h, __doc__)
