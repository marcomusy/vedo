##########################################################
# Morph one shape into another using spherical harmonics
# package shtools.
# In this example we morph a sphere into a octahedron
# and viceversa
##########################################################
from __future__ import division, print_function
try:
    import pyshtools
except:
    print('Please install pyshtools to run this example')
    print('Follow instructions at https://shtools.oca.eu/shtools')
    exit(0)
    
from vtkplotter import Plotter, mag, arange, sphere, sin, cos
import numpy as np

##########################################################
N = 100      # number of sample points on the unit sphere
lmax = 15    # maximum degree of the sph. harm. expansion
rbias = 0.5  # subtract a constant average value
x0 = [0,0,0] # set object at this position
##########################################################

def makegrid(shape, N):
    rmax = 2.0   # line length 
    agrid, pts = [], []
    for th in np.linspace(0, np.pi, N, endpoint=True):
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
    actor = vp.points(pts, c='k', alpha=0.4, r=1)
    return agrid, actor

def morph(clm1, clm2, t, lmax):
    # interpolate linearly the two sets of sph harm. coeeficients
    clm = (1-t) * clm1 + t * clm2
    grid_reco = clm.expand(lmax=lmax) # cut "high frequency" components
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
    return pts


vp = Plotter(shape=[2,2], verbose=0, axes=3, interactive=0)

shape1 = sphere(alpha=0.2)
shape2 = vp.load('data/shapes/icosahedron.vtk').normalize().lineWidth(1)

agrid1, actorpts1 = makegrid(shape1, N)
vp.show(at=0, actors=[shape1, actorpts1])

agrid2, actorpts2 = makegrid(shape2, N)
vp.show(at=1, actors=[shape2, actorpts2])
vp.camera.Zoom(1.2)
vp.interactive = False

clm1  = pyshtools.SHGrid.from_array(agrid1).expand()
clm2  = pyshtools.SHGrid.from_array(agrid2).expand()
# clm1.plot_spectrum2d() # plot the value of the sph harm. coefficients
# clm2.plot_spectrum2d() 

for t in arange(0,1, 0.005):
    act21 = vp.points(morph(clm2, clm1, t, lmax), c='r', r=4)
    act12 = vp.points(morph(clm1, clm2, t, lmax), c='g', r=4)
    vp.show(at=2, actors=act21, resetcam=0, legend='time: '+str(int(t*100)))
    vp.show(at=3, actors=act12)
    vp.camera.Azimuth(2)

vp.show(interactive=1)