"""Erode or dilate a Volume
by replacing a voxel with the max/min
over an ellipsoidal neighborhood"""
from vedo import *

embryo = Volume(dataurl+'embryo.tif')
embryo.printHistogram(logscale=1)

eroded = embryo.clone().erode(neighbours=(2,2,2))
dilatd = eroded.clone().dilate(neighbours=(2,2,2))

show([(embryo, __doc__), eroded, dilatd], N=3, viewup='z', zoom=1.4).close()

