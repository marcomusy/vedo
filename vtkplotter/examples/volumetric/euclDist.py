"""Euclidean Distance Transform using Saito algorithm.
The distance map produced contains the square
of the Euclidean distance values.
"""
from vtkplotter import *

t = Text2D(__doc__, c='white')
e = load(datadir+'embryo.tif') # Volume

edt = e.euclideanDistance()

show([(e,t), edt], N=2, viewup='z', axes=1, zoom=1.5)

