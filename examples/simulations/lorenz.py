"""The Lorenz attractor is a set of chaotic solutions of
a particular system of ordinary differential equations"""
from vedo import *

p = (25.0, -10.0, -7.0)  # starting point (initial condition)
dt = 0.002

pts, vel = [], []
for t in np.arange(0, 20, dt):
    x, y, z = p
    dpdt = [-8/3 * x + y*z,  -10*(y-z),  -y*x + 28*y-z]
    p = p + np.array(dpdt) * dt
    pts.append(p)
    vel.append(mag(dpdt))

line = Line(pts).lw(3).cmap("winter", vel)
line.add_scalarbar("speed")
line.add_shadow("x",   3, alpha=0.2)
line.add_shadow("z", -25, alpha=0.2)

plt = Plotter(axes=dict(xygrid=False))
plt.show(__doc__, line, viewup="z").close()
