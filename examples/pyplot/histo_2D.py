"""Histogram of 2 variables"""
from vedo import *
from vedo.pyplot import histogram
import numpy as np

n = 10000
x = np.random.normal(2, 1, n)*2 + 3
y = np.random.normal(1, 1, n)*1 + 7
xm, ym = np.mean(x), np.mean(y)

h = histogram(x, y,
              bins=50,
              aspect=4/3,
              cmap='Blues',
              title='2D Gauss histo',
              scalarbar=True,
             )

# add some object to the plot
h += Marker('*', s=0.3, c='r').pos(xm, ym, 0.1)

show(h, mode="image").close()
