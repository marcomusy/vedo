"""A uniform distribution on a plane
is not uniform on a sphere"""
import numpy as np
from vedo.pyplot import histogram
from vedo import show

phi = np.random.rand(1000)*6.28
the = np.random.rand(1000)*3.14

h = histogram(the, phi, mode='spheric').add_scalarbar()

show(h, __doc__, axes=12, viewup='z').close()
