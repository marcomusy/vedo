#!/usr/bin/python3
#
"""Interpolate a jpg image to a mesh and plot it"""
from dolfin import *
import matplotlib.pyplot as plt
from vedo.dolfin import plot, download

scale = 0.1
fpath = download("https://vedo.embl.es/examples/data/images/embl_logo.jpg")
img = plt.imread(fpath)
print('Image shape is', img.shape)

img = img[:,:,1]
Nx, Ny = img.shape
mesh = RectangleMesh(Point(0,0,0),
                     Point(Ny*scale, Nx*scale,1), Ny, Nx)

class FE_image(UserExpression):
    def eval_cell(self, value, x, ufc_cell):
        p = Cell(mesh, ufc_cell.index).midpoint()
        i, j = int(p[1]/scale), int(p[0]/scale)
        value[:] = img[-(i+1), j]
    def value_shape(self):
        return ()

y = FE_image()
V = FunctionSpace(mesh, 'Lagrange', 1)
u = Function(V)
u.interpolate(y)

cam = dict(pos=(10.6, 3.71, 22.7),
           focalPoint=(10.6, 3.71, -1.04e-3),
           viewup=(0, 1.00, 0),
           distance=22.7,
           clippingRange=(21.3, 24.6)) # press C to get this lines of code

plot(u,
     text=__doc__,
     camera=cam,
     lw=0.1,
     cmap='Greens_r',
     size=(600,300)
     )