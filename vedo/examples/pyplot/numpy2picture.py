"""Load a picture with matplotlib imread()
and make it a 3D object.
"""
from matplotlib.image import imread
from vedo import *

fname = datadir+'images/harvest.jpg'
arr = imread(fname)

pic = Picture(arr) # create Picture object from numpy array

# (press r to reset):
show(pic, __doc__, axes=8)
