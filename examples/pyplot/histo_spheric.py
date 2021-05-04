"""A uniform distribution on a plane
is not uniform on a sphere"""
import numpy as np
from vedo.pyplot import histogram

phi = np.random.rand(1000)*6.28
the = np.random.rand(1000)*3.14

h = histogram(the, phi, mode='spheric')

h.show(axes=12, viewup='z').close()
