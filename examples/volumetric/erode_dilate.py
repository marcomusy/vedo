"""Erode or dilate a Volume
by replacing a voxel with the max/min
over an ellipsoidal neighborhood"""
from vedo import *

em = Volume(dataurl+'embryo.tif').lighting('plastic').printHistogram(logscale=1)

eroded = em.clone().erode(neighbours=(5,5,5))
dilatd = em.clone().dilate(neighbours=(5,5,5))

show([(em, __doc__), eroded, dilatd], N=3, viewup='z', zoom=1.4)

