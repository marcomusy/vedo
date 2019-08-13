"""Load a picture with matplotlib imread()
and make it a 3D object.
"""
from matplotlib.image import imread
from vtkplotter import Picture, Text, datadir, show

fname = datadir+'images/tropical.jpg'
arr = imread(fname)
print('loaded',fname, '\ntype is:',type(arr), arr.shape)

pic = Picture(arr) # Create Picture from numpy array

show(pic, Text(__doc__))
