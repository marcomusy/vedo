"""A uniform distribution on a plane
is not uniform on a sphere"""
import numpy as np
from vedo.pyplot import histogram
from vedo import Plotter

phi = np.random.rand(1000)*np.pi*2
the = np.random.rand(1000)*np.pi

h = histogram(the, phi, mode='spheric').add_scalarbar()

plt = Plotter(axes=12).add_ambient_occlusion(0.05)
plt.show(h, __doc__, viewup='z').close()
