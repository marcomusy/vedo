'''
Example on how to use the intersectWithLine() method:
 intersect an actor with lines from the origin
 and draw the intersection points in blue

Second part of the example:
 Expand an arbitrary closed shape in spherical harmonics 
 using SHTOOLS (https://shtools.oca.eu/shtools/)
 and then truncate the expansion to a specific lmax and 
 reconstruct the projected points in red
'''
from __future__ import division, print_function
print(__doc__)

from vtkplotter import Plotter, Points, mag, sin, cos
import numpy as np

##########################################################
N = 100      # number of sample points on the unit sphere
lmax = 15    # maximum degree of the expansion
rmax = 2.0   # line length 
rbias = 0.5  # subtract a constant average value
x0 = [0,0,0] # set object at this position
##########################################################

vp = Plotter(shape=[1,2], verbose=0, axes=0)
shape = vp.load('data/shapes/icosahedron.vtk').normalize().pos(x0).lineWidth(2)

agrid, pts = [], []
for th in np.linspace(0, np.pi,   N, endpoint=True):
    lats = []
    for ph in np.linspace(0, 2*np.pi, N, endpoint=True):
        p  = np.array([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])*rmax
        intersections = shape.intersectWithLine([0,0,0], p) ### <--
        if len(intersections):
            value = mag(intersections[0])
            lats.append(value - rbias)
            pts.append(intersections[0])
        else:
            lats.append(rmax - rbias)
            pts.append(p)
    agrid.append(lats)
agrid = np.array(agrid)

vp.add(Points(pts, c='b', r=2))
vp.show(at=0)

############################################################
try:
    import pyshtools
except:
    print('Please install pyshtools to run this example')
    print('Follow instructions at https://shtools.oca.eu/shtools')
    exit(0)

grid = pyshtools.SHGrid.from_array(agrid)
grid.plot() # plots the scalars in a 2d plots latitudes vs longitudes

clm = grid.expand()
clm.plot_spectrum2d() # plot the value of the sph harm. coefficients

grid_reco = clm.expand(lmax=lmax) # cut "high frequency" components
grid_reco.plot()
agrid_reco = grid_reco.to_array()
pts = []
for i, longs in enumerate(agrid_reco):
    ilat = grid_reco.lats()[i]
    for j, value in enumerate(longs):
        ilong = grid_reco.lons()[j]
        th = (90 - ilat)/57.3
        ph = ilong/57.3
        r  = value + rbias
        p  = np.array([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])*r
        pts.append(p)

act = Points(pts, c='r', r=8, alpha=0.5)
vp.show(at=1, actors=act, interactive=1)

