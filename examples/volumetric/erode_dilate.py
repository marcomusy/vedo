"""Erode or dilate a Volume
by replacing a voxel with the max/min
over an ellipsoidal neighborhood.
"""
from vtkplotter import *

t = Text(__doc__, c='white')
e = load(datadir+'embryo.slc').printHistogram(logscale=1)

eroded = erodeVolume( e, neighbours=(5,5,5)).printHistogram(logscale=1)
dilatd = dilateVolume(e, neighbours=(5,5,5)).printHistogram(logscale=1)

show([(e, t), eroded, dilatd], N=3, viewup='z', zoom=2)

