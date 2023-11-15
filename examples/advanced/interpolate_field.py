"""Interpolate a vectorial field using
Thin Plate Spline or Radial Basis Function"""
from scipy.interpolate import Rbf
from vedo import Plotter, Points, Arrows, show
import numpy as np


ls = np.linspace(0, 10, 8)
X, Y, Z = np.meshgrid(ls, ls, ls)
xr, yr, zr = X.ravel(), Y.ravel(), Z.ravel()
positions = np.vstack([xr, yr, zr]).T

sources = [(5, 8, 5), (8, 5, 5), (5, 2, 5)]
deltas = [(1, 1, 0.2), (1, 0, -0.8), (1, -1, 0.2)]

apos = Points(positions, r=2)

# for p in apos.vertices: ####### Uncomment to fix some points.
#    if abs(p[2]-5) > 4.999:  # differences btw RBF and thinplate
#        sources.append(p)    # will become much smaller.
#        deltas.append(np.zeros(3))
sources = np.array(sources)
deltas = np.array(deltas)

src = Points(sources).color("r").ps(12)
trs = Points(sources + deltas).color("v").ps(12)
arr = Arrows(sources, sources + deltas).color("k8")

################################################# warp using Thin Plate Splines
warped = apos.clone().warp(sources, sources+deltas)
warped.alpha(0.4).color("lg").point_size(10)
allarr = Arrows(apos.vertices, warped.vertices).color("k8")

set1 = [apos, warped, src, trs, arr, __doc__]
plt1 = Plotter(N=2, bg='bb')
plt1.at(0).show(apos, warped, src, trs, arr, __doc__)
plt1.at(1).show(allarr)


################################################# RBF
x, y, z = sources[:, 0], sources[:, 1], sources[:, 2]
dx, dy, dz = deltas[:, 0], deltas[:, 1], deltas[:, 2]

itrx = Rbf(x, y, z, dx)  # Radial Basis Function interpolator:
itry = Rbf(x, y, z, dy)  #  interoplate the deltas in each separate
itrz = Rbf(x, y, z, dz)  #  cartesian dimension

positions_x = itrx(xr, yr, zr) + xr
positions_y = itry(xr, yr, zr) + yr
positions_z = itrz(xr, yr, zr) + zr
positions_rbf = np.vstack([positions_x, positions_y, positions_z]).T

warped_rbf = Points(positions_rbf).color("lg",0.4).point_size(10)
allarr_rbf = Arrows(apos.vertices, warped_rbf.vertices).color("k8")

arr = Arrows(sources, sources + deltas).color("k8")

plt2 = Plotter(N=2, pos=(200, 300), bg='bb')
plt2.at(0).show("Radial Basis Function", apos, warped_rbf, src, trs, arr)
plt2.at(1).show(allarr_rbf)
plt2.interactive()

plt2.close()
plt1.close()

