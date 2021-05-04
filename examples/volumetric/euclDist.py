"""Euclidean Distance Transform using Saito algorithm.
The distance map produced contains the square
of the Euclidean distance values"""
from vedo import *

e = Volume(dataurl+'embryo.tif')

edt = e.euclideanDistance()

show([(e,__doc__), edt], N=2, viewup='z', axes=1, zoom=1.5).close()

