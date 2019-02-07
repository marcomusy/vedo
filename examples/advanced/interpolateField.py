'''
Interpolate a vectorial field using Thin Plate Spline or Radial Basis Function.
Example shows how to share the same vtkCamera between different Plotter windows.
'''
print(__doc__)
from vtkplotter import Plotter, thinPlateSpline, points, arrows, show, text
import numpy as np

n=8j # make a grid n.n.n
X, Y, Z = np.mgrid[0:10:n, 0:10:n, 0:10:n]
xr, yr, zr = X.ravel(), Y.ravel(), Z.ravel()
positions = np.vstack([xr, yr, zr])

sources = [
        (5,8,5),
        (8,5,5),
        (5,2,5),
        ]
deltas = [
        (1, 1,.2),
        (1,0,-.8),
        (1,-1,.2),
        ]

apos = points(positions, r=2)

#for p in apos.coordinates(): ####### Uncomment to fix some points.
#    if abs(p[2]-5) > 4.999:  # differences btw RBF and thinplate
#        sources.append(p)    # will become much smaller.
#        deltas.append(np.zeros(3))
sources = np.array(sources)
deltas  = np.array(deltas)

src = points(sources, c='r', r=12)
trs = points(sources+deltas, c='v', r=12)
arr = arrows(sources, sources+deltas)

################################################# Thin Plate Splines
warped = thinPlateSpline(apos, sources, sources+deltas)
warped.alpha(0.4).color('lg').pointSize(10)
allarr = arrows(apos.coordinates(), warped.coordinates())

set1 = [apos, warped, src, trs, arr, text("Thin Plate Splines")]
vp = show([set1, allarr], N=2, verbose=0) # returns the Plotter class


################################################# RBF
from scipy.interpolate import Rbf

x,y,z    = sources[:,0], sources[:,1], sources[:,2]
dx,dy,dz = deltas[:,0],  deltas[:,1],  deltas[:,2]

itrx = Rbf(x,y,z, dx) # Radial Basis Function interpolator:
itry = Rbf(x,y,z, dy) #  interoplate the deltas in each separate 
itrz = Rbf(x,y,z, dz) #  cartesian dimension

positions_x = itrx(xr, yr, zr) + xr
positions_y = itry(xr, yr, zr) + yr
positions_z = itrz(xr, yr, zr) + zr
positions_rbf = np.vstack([positions_x, positions_y, positions_z])

warped_rbf = points(positions_rbf, r=2).alpha(0.4).color('lg').pointSize(10)
allarr_rbf = arrows(apos.coordinates(), warped_rbf.coordinates())

arr = arrows(sources, sources+deltas)

vp2 = Plotter(N=2, pos=(200,300), verbose=0)
vp2.camera = vp.camera # share the same camera with previous Plotter
vp2.show([apos, warped_rbf, src, trs, arr, text("Radial Basis Function")], at=0)
vp2.show(allarr_rbf, at=1, interactive=1)









