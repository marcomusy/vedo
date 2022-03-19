"""Generate a time sequence of 3D shapes
(from a sphere to a tetrahedron) as noisy point clouds,
and smooth it with Moving Least Squares (smoothMLS3D).
This make a simultaneus fit in 4D (space+time).
smoothMLS3D method returns points that
are color coded in bins of fitted time.
Data itself can suggest a meaningful time separation
based on the spatial distribution of points"""
from vedo import *

dt = 0.1                     # space to time unit conversion
neighbours = 50              # nr neighbours in the local 4D search
a, b, noise = 0.2, 0.4, 1.0  # some random warping parameters and noise level

# generate uniform points on sphere
# (tol separates points by 1.5% of mesh size)
ss = Sphere(res=200).subsample(0.015)
cc = ss.points()

sets, warps = [], []
for i in range(5):  # generate a time sequence of 5 shapes
    cs = cc + a * i * cc**2 + b * i * cc**3  # warp sphere in weird ways
    # set absolute time of points, and add noise to positions
    ap = Points(cs).c(i).addGaussNoise(noise).time(dt*i)
    sets.append(ap)
    ssc = ss.clone().points(cs).wireframe().c('gray').alpha(.01)
    warps.append(ssc)

plt = Plotter(N=3, size=(1600,700))
plt.at(0).show(warps, Text2D(__doc__, s=0.75))
plt.at(1).show(sets, "add noise to vertices:", zoom=1.4)

#The nr neighbours in the local 4D search must be specified
sm3d = smoothMLS3D(sets, neighbours)

#color indicates fitted time
sm3d.addScalarBar(title='time [a.u.]')

plt.at(2).show(sm3d, "4D-smoothed", zoom=1.4)
plt.interactive().close()

############################### compare generated+noise to smoothed
plt3 = Plotter(shape=(2,5))
for i in range(5):
    plt3.at(i).show(warps, sets[i], "t="+str(i))
    t0, t1 = (i-0.5)*dt, (i+0.5)*dt
    sm3d1 = sm3d.clone().threshold("PointScalars", t0, t1).alpha(1)
    msg = "time range = ["+precision(t0, 2)+","+precision(t1, 2)+"]"
    plt3.at(i+5).show(warps, sm3d1, msg)
plt3.interactive().close()

############################### make a video
# vd = Video('mls3d.mp4')
plt2 = Plotter(interactive=False)
plt2.show(warps, zoom=1.7)
for j in range(100):
    i = j/20
    t0, t1 = (i-0.5)*dt, (i+0.5)*dt
    sm3d1 = sm3d.clone().threshold("PointScalars", t0, t1).alpha(1)
    msg = "time range = ["+precision(t0, 2)+","+precision(t1, 2)+"]"
    plt2.show(warps, sm3d1, msg, resetcam=0)
    # vd.addFrame()
    plt2.clear()
# vd.close()

plt2.interactive().close()

