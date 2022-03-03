"""
Morph one shape into another using spherical harmonics package shtools.

In this example we morph a sphere into a octahedron and viceversa.
"""
import numpy as np
from vedo import settings, Plotter, Points, Sphere, cos, dataurl, mag, sin

try:
    import pyshtools
    print(__doc__)
except ModuleNotFoundError:
    print("Please install pyshtools to run this example")
    print("Follow instructions at https://shtools.oca.eu/shtools")
    exit(0)


##########################################################
N = 100  # number of sample points on the unit sphere
lmax = 15  # maximum degree of the sph. harm. expansion
rbias = 0.5  # subtract a constant average value
x0 = [0, 0, 0]  # set object at this position
##########################################################


def makeGrid(shape, N):
    rmax = 2.0  # line length
    agrid, pts = [], []
    for th in np.linspace(0, np.pi, N, endpoint=True):
        lats = []
        for ph in np.linspace(0, 2 * np.pi, N, endpoint=True):
            p = np.array([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]) * rmax
            intersections = shape.intersectWithLine([0, 0, 0], p)
            if len(intersections):
                value = mag(intersections[0])
                lats.append(value - rbias)
                pts.append(intersections[0])
            else:
                lats.append(rmax - rbias)
                pts.append(p)
        agrid.append(lats)
    agrid = np.array(agrid)
    actor = Points(pts, c="k", alpha=0.4, r=1)
    return agrid, actor


def morph(clm1, clm2, t, lmax):
    """Interpolate linearly the two sets of sph harm. coeeficients."""
    clm = (1 - t) * clm1 + t * clm2
    grid_reco = clm.expand(lmax=lmax)  # cut "high frequency" components
    agrid_reco = grid_reco.to_array()
    pts = []
    for i, longs in enumerate(agrid_reco):
        ilat = grid_reco.lats()[i]
        for j, value in enumerate(longs):
            ilong = grid_reco.lons()[j]
            th = np.deg2rad(90 - ilat)
            ph = np.deg2rad(ilong)
            r = value + rbias
            p = np.array([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]) * r
            pts.append(p)
    return pts

settings.useDepthPeeling = True

plt = Plotter(shape=[2, 2], axes=3, interactive=0)

shape1 = Sphere(alpha=0.2)
shape2 = plt.load(dataurl + "icosahedron.vtk").normalize().lineWidth(1)

agrid1, actorpts1 = makeGrid(shape1, N)

plt.at(0).show(shape1, actorpts1)

agrid2, actorpts2 = makeGrid(shape2, N)
plt.at(1).show(shape2, actorpts2)

plt.camera.Zoom(1.2)

clm1 = pyshtools.SHGrid.from_array(agrid1).expand()
clm2 = pyshtools.SHGrid.from_array(agrid2).expand()
# clm1.plot_spectrum2d()  # plot the value of the sph harm. coefficients
# clm2.plot_spectrum2d()

for t in np.arange(0, 1, 0.005):
    act21 = Points(morph(clm2, clm1, t, lmax), c="r", r=4)
    act12 = Points(morph(clm1, clm2, t, lmax), c="g", r=4)

    plt.at(2).show(act21, resetcam=False)
    plt.at(3).show(act12)
    plt.camera.Azimuth(2)

plt.interactive().close()
