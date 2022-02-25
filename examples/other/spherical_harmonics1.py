"""Expand and reconstruct any surface
(here a simple box) into spherical harmonics"""
# Expand an arbitrary closed shape in spherical harmonics
# using SHTOOLS (https://shtools.oca.eu/shtools/)
# and then truncate the expansion to a specific lmax and
# reconstruct the projected points in red
import numpy as np
from scipy.interpolate import griddata
import pyshtools
from vedo import spher2cart, mag, Box, Point, Points, show

###########################################################################
lmax = 8              # maximum degree of the spherical harm. expansion
N    = 50             # number of grid intervals on the unit sphere
rmax = 500            # line length
x0 = [250, 250, 250]  # set SPH sphere at this position
###########################################################################

x0 = np.array(x0)
surface = Box(pos=x0+[10,20,30], size=(300,150,100)).color('grey').alpha(0.2)

############################################################
# cast rays from the sphere center and find intersections
agrid, pts = [], []
for th in np.linspace(0, np.pi, N, endpoint=True):
    longs = []
    for ph in np.linspace(0, 2*np.pi, N, endpoint=False):
        p = spher2cart(rmax, th, ph)
        intersections = surface.intersectWithLine(x0, x0+p)
        if len(intersections):
            value = mag(intersections[0]-x0)
            longs.append(value)
            pts.append(intersections[0])
        else:
            print('No hit for theta, phi =', th, ph, c='r')
            longs.append(rmax)
            pts.append(p)
    agrid.append(longs)
agrid = np.array(agrid)

hits = Points(pts).cmap('jet', agrid.ravel()).addScalarBar3D(title='scalar distance to x_0')
show([surface, hits, Point(x0), __doc__], at=0, N=2, axes=1)

#############################################################
grid = pyshtools.SHGrid.from_array(agrid)
clm = grid.expand()
grid_reco = clm.expand(lmax=lmax).to_array()  # cut "high frequency" components

#############################################################
# interpolate to a finer grid
ll = []
for i, long in enumerate(np.linspace(0, 360, num=grid_reco.shape[1], endpoint=False)):
    for j, lat in enumerate(np.linspace(90, -90, num=grid_reco.shape[0], endpoint=True)):
        th = np.deg2rad(90 - lat)
        ph = np.deg2rad(long)
        p = spher2cart(grid_reco[j][i], th, ph)
        ll.append((lat, long))

radii = grid_reco.T.ravel()
n = 200j
lnmin, lnmax = np.array(ll).min(axis=0), np.array(ll).max(axis=0)
grid = np.mgrid[lnmax[0]:lnmin[0]:n, lnmin[1]:lnmax[1]:n]
grid_x, grid_y = grid
grid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='cubic')

pts2 = []
for i, long in enumerate(np.linspace(0, 360, num=grid_reco_finer.shape[1], endpoint=False)):
    for j, lat in enumerate(np.linspace(90, -90, num=grid_reco_finer.shape[0], endpoint=True)):
        th = np.deg2rad(90 - lat)
        ph = np.deg2rad(long)
        p = spher2cart(grid_reco_finer[j][i], th, ph)
        pts2.append(p+x0)

show(f'Spherical harmonics expansion of order {lmax}',
     Points(pts2, c="r", alpha=0.5),
     surface,
     at=1,
).interactive().close()

