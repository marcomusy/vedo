"""Euclidean Distance Transform using Saito algorithm.
The distance map produced contains the square
of the Euclidean distance values.
"""
from vtkplotter import *

t = Text(__doc__, c='white')
e = load(datadir+'embryo.tif') # Volume

edt = euclideanDistanceVolume(e)

show([(e,t), edt], N=2, viewup='z', zoom=1.5)

