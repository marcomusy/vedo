"""
Example on how to use the intersectWithLine() method:
 intersect an actor with lines from the origin
 and draw the intersection points in blue

Second part of the example:
 Expand an arbitrary closed shape in spherical harmonics
 using SHTOOLS (https://shtools.oca.eu/shtools/)
 and then truncate the expansion to a specific lmax and
 reconstruct the projected points in red
"""
from __future__ import division, print_function
import numpy as np
from scipy.interpolate import griddata
from vtkplotter import show, load, Points, datadir, mag, spher2cart
print(__doc__)


#############################################################
N = 30          # number of grid intervals on the unit sphere
lmax = 15       # maximum degree of the expansion
rmax = 2.0      # line length
x0 = [0, 0, 0]  # set object at this position
#############################################################

shape = load(datadir + "pumpkin.vtk").normalize().pos(x0).lineWidth(.1)

show(shape, at=0, N=2, bg='w')


############################################################
# cast rays from the center and find intersections
agrid, pts = [], []
for th in np.linspace(0, np.pi, N, endpoint=True):
    lats = []
    for ph in np.linspace(0, 2*np.pi, N, endpoint=True):
        p = spher2cart(rmax, th, ph)
        intersections = shape.intersectWithLine([0, 0, 0], p)  # <--------------
        if len(intersections):
            value = mag(intersections[0])
            lats.append(value)
            pts.append(intersections[0])
        else:
            lats.append(rmax)
            pts.append(p)
    agrid.append(lats)
agrid = np.array(agrid)


############################################################
# Please install pyshtools to continue this example
# Follow instructions at https://shtools.oca.eu/shtools
import pyshtools   

grid = pyshtools.SHGrid.from_array(agrid)
clm = grid.expand()
grid_reco = clm.expand(lmax=lmax)  # cut "high frequency" components

#grid.plot()  # plots the scalars in a 2d plots latitudes vs longitudes
#clm.plot_spectrum2d()  # plot the value of the sph harm. coefficients
#grid_reco.plot()

agrid_reco = grid_reco.to_array()
pts1 = []
ll = []
for i, longs in enumerate(agrid_reco):
    ilat = grid_reco.lats()[i]
    for j, r in enumerate(longs):
        ilong = grid_reco.lons()[j]
        th = np.deg2rad(90 - ilat)
        ph = np.deg2rad(ilong)
        p = spher2cart(r, th, ph)
        pts1.append(p)
        ll.append((ilat, ilong))

radii = agrid_reco.ravel()

#############################################################
# make a finer grid

#ll.append((-90, 360))
#radii = np.append(radii, radii[0])
#print(agrid_reco.shape)
#print(np.array(ll).min(axis=0))
#print(np.array(ll).max(axis=0))

n = 200j
lmin, lmax = np.array(ll).min(axis=0), np.array(ll).max(axis=0)
grid = np.mgrid[lmin[0]:lmax[0]:n, lmin[1]:lmax[1]:n]
grid_x, grid_y = grid
agrid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='cubic')

pts2 = []
for i, lat in enumerate(grid_x[:,1]):
    for j, long in enumerate(grid_y[0]):
        th = np.deg2rad(90 -lat)
        ph = np.deg2rad(long)
        r = agrid_reco_finer[i][j]
        p = spher2cart(r, th, ph)
        pts2.append(p)

act1 = Points(pts1, r=8, c="b", alpha=0.5)
act2 = Points(pts2, r=2, c="r", alpha=0.5)

show(act1, act2, at=1, interactive=1)



















