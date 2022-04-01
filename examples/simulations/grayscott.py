# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/gray_scott.py
# Parameters from http://www.aliensaint.com/uo/java/rd
# Adapted for vedo by Marco Musy (2020)
# -----------------------------------------------------------------------------
"""Grey-Scott reaction-diffusion system"""
import numpy as np
from vedo import Plotter, Grid, settings

# ---------------------------------------------------------------
Nsteps = 300
n = 200 # grid subdivisions
#Du, Dv, F, k, name = 0.16, 0.08, 0.035, 0.065, 'Bacteria 1'
#Du, Dv, F, k, name = 0.14, 0.06, 0.035, 0.065, 'Bacteria 2'
#Du, Dv, F, k, name = 0.16, 0.08, 0.060, 0.062, 'Coral'
#Du, Dv, F, k, name = 0.19, 0.05, 0.060, 0.062, 'Fingerprint'
#Du, Dv, F, k, name = 0.10, 0.10, 0.018, 0.050, 'Spirals'
#Du, Dv, F, k, name = 0.12, 0.08, 0.020, 0.050, 'Spirals Dense'
#Du, Dv, F, k, name = 0.10, 0.16, 0.020, 0.050, 'Spirals Fast'
#Du, Dv, F, k, name = 0.16, 0.08, 0.020, 0.055, 'Unstable'
#Du, Dv, F, k, name = 0.16, 0.08, 0.050, 0.065, 'Worms 1'
#Du, Dv, F, k, name = 0.16, 0.08, 0.054, 0.063, 'Worms 2'
Du, Dv, F, k, name = 0.16, 0.08, 0.035, 0.060, 'Zebrafish'
# ---------------------------------------------------------------

settings.allowInteraction = True

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
grd = Grid(s=[sx,sy], res=[sx,sy])
grd.lineWidth(0).wireframe(False).lighting(ambient=0.5)
formula = r'(u,v)=(D_u\cdot\Delta u -u v v+F(1-u), D_v\cdot\Delta v +u v v -(F+k)v)'
print('Du, Dv, F, k, name =', Du, Dv, F, k, name)

plt = Plotter(bg='linen', interactive=False)
plt.show(grd, __doc__, zoom=1.25, elevation=-30)

for step in range(Nsteps):
    for i in range(25):
        Lu = (                  U[0:-2, 1:-1] +
              U[1:-1, 0:-2] - 4*U[1:-1, 1:-1] + U[1:-1, 2:] +
                                U[2:  , 1:-1])
        Lv = (                  V[0:-2, 1:-1] +
              V[1:-1, 0:-2] - 4*V[1:-1, 1:-1] + V[1:-1, 2:] +
                                V[2:  , 1:-1])
        uvv = u*v*v
        u += Du*Lu - uvv + F*(1-u)
        v += Dv*Lv + uvv - (F+k)*v

    grd.cmap('ocean_r', V.ravel(), on='cells', name="escals")
    grd.mapCellsToPoints()                   # interpolate cell data to point data
    newpts = grd.points()
    newpts[:,2] = grd.pointdata['escals']*25 # assign z elevation
    grd.points(newpts)                       # set the new points

    plt.render()
    if plt.escaped:
        break  # if ESC is hit during loop

plt.interactive().close()
