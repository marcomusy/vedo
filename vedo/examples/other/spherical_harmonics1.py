"""
 Expand an arbitrary closed shape in spherical harmonics
 using SHTOOLS (https://shtools.oca.eu/shtools/)
 and then truncate the expansion to a specific lmax and
 reconstruct the projected points in red
"""
import numpy as np

# Please install pyshtools to run this example
# Follow instructions at https://shtools.oca.eu/shtools
import pyshtools

from scipy.interpolate import griddata
from vedo import Points, load, mag, show, spher2cart, datadir
print(__doc__)

#############################################################
lmax = 10       # maximum degree of the spherical harm. expansion
N = 50          # number of grid intervals on the unit sphere
rmax = 1.5      # line length
x0 = [0, 0, 0]  # set object at this position
#############################################################

shape = load(datadir+'apple.ply').normalize().pos(x0).lineWidth(0.1)

show(shape, at=0, N=2, axes=dict(zxGrid=False))

############################################################
# cast rays from the center and find intersections
agrid, pts = [], []
for th in np.linspace(0, np.pi, N, endpoint=False):
    lats = []
    for ph in np.linspace(0, 2*np.pi, N, endpoint=False):
        p = spher2cart(rmax, th, ph)
        intersections = shape.intersectWithLine([0, 0, 0], p)
        if len(intersections):
            value = mag(intersections[0])
            lats.append(value)
            pts.append(intersections[0])
        else:
            lats.append(rmax)
            pts.append(p)
    agrid.append(lats)
agrid = np.array(agrid)

grid = pyshtools.SHGrid.from_array(agrid)
clm = grid.expand()
grid_reco = clm.expand(lmax=lmax)  # cut "high frequency" components


#############################################################
# interpolate to a finer grid
agrid_reco = grid_reco.to_array()

ll = []
for i, long in enumerate(np.linspace(0, 360, num=agrid_reco.shape[1], endpoint=True)):
    for j, lat in enumerate(np.linspace(90, -90, num=agrid_reco.shape[0], endpoint=True)):
        th = np.deg2rad(90 - lat)
        ph = np.deg2rad(long)
        p = spher2cart(agrid_reco[j][i], th, ph)
        ll.append((lat, long))

radii = agrid_reco.T.ravel()
n = 200j
lnmin, lnmax = np.array(ll).min(axis=0), np.array(ll).max(axis=0)
grid = np.mgrid[lnmax[0]:lnmin[0]:n, lnmin[1]:lnmax[1]:n]
grid_x, grid_y = grid
agrid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='cubic')

pts2 = []
for i, long in enumerate(np.linspace(0, 360, num=agrid_reco_finer.shape[1], endpoint=False)):
    for j, lat in enumerate(np.linspace(90, -90, num=agrid_reco_finer.shape[0], endpoint=True)):
        th = np.deg2rad(90 - lat)
        ph = np.deg2rad(long)
        p = spher2cart(agrid_reco_finer[j][i], th, ph)
        pts2.append(p)

mesh2 = Points(pts2, r=2, c="r", alpha=0.5)
mesh2.clean(0.01) # impose point separation of 1% of the bounding box size

show(mesh2,
     'Spherical harmonics\nexpansion of order '+str(lmax),
     at=1, interactive=True)
