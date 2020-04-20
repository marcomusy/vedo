# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/gray_scott.py
# Parameters from http://www.aliensaint.com/uo/java/rd
# Adapted for vtkplotter by Marco Musy (2020)
# -----------------------------------------------------------------------------
import numpy as np
from vtkplotter import Grid, interactive

# -----------------------------------------------------
Nsteps = 300
n = 200 # grid subdivisions
# Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065  # Bacteria 1
# Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065  # Bacteria 2
# Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062  # Coral
# Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062  # Fingerprint
# Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050  # Spirals
# Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050  # Spirals Dense
# Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050  # Spirals Fast
# Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055  # Unstable
# Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065  # Worms 1
# Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063  # Worms 2
Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060  # Zebrafish
# -----------------------------------------------------


Z = np.zeros((n+2, n+2), [('U', np.double), ('V', np.double)])
U, V = Z['U'], Z['V']
u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]

r = 20
u[...] = 1.0
U[n//2-r:n//2+r, n//2-r:n//2+r] = 0.50
V[n//2-r:n//2+r, n//2-r:n//2+r] = 0.25
u += 0.05*np.random.uniform(-1, 1, (n, n))
v += 0.05*np.random.uniform(-1, 1, (n, n))

sy, sx = V.shape
grd = Grid(sx=sx, sy=sy, resx=sx, resy=sy)
grd.lineWidth(0).wireframe(False).lighting(ambient=0.5)

for step in range(Nsteps):
    for i in range(20):
        Lu = (                  U[0:-2, 1:-1] +
              U[1:-1, 0:-2] - 4*U[1:-1, 1:-1] + U[1:-1, 2:] +
                                U[2:  , 1:-1])
        Lv = (                  V[0:-2, 1:-1] +
              V[1:-1, 0:-2] - 4*V[1:-1, 1:-1] + V[1:-1, 2:] +
                                V[2:  , 1:-1])
        uvv = u*v*v
        u += (Du*Lu - uvv + F*(1-u))
        v += (Dv*Lv + uvv - (F+k)*v)

    vvals = np.flip(V, axis=0).ravel()
    grd.cellColors(vvals, cmap='ocean_r').mapCellsToPoints()
    newpts = np.c_[grd.points()[:,[0,1]], grd.getPointArray('cellColors')*20]
    grd.points(newpts).show(axes=9, zoom=1.3, elevation=-.15, interactive=False)

interactive()
