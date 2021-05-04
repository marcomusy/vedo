"""Load a picture with matplotlib imread()
and make it a 3D object"""
from matplotlib.image import imread
from vedo import *

fname = download('https://vedo.embl.es/examples/data/images/tropical.jpg')

arr = imread(fname)

pic = Picture(arr) # create Picture object from numpy array

show(pic, __doc__, axes=7).close()
