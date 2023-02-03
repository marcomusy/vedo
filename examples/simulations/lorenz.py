"""The Lorenz attractor is a set of chaotic solutions of
a particular system of ordinary differential equations"""
from vedo import *

dt = 0.002
y = (25.0, -10.0, -7.0)  # Starting point (initial condition)
pts, vel = [], []

for t in np.linspace(0, 20, int(20 / dt)):
    # Integrate a funny differential equation
    dydt = np.array(
        [-8 / 3.0 * y[0] + y[1] * y[2],
         -10.0 * (y[1] - y[2]),
         -y[1] * y[0] + 28.0 * y[1] - y[2]]
    )
    y = y + dydt * dt
    v = np.clip([np.linalg.norm(dydt) * 0.005], 0, 1)[0]  # color by speed
    vel.append(v)
    pts.append(y)

line = Line(pts).lw(3).cmap("brg", vel)
line.add_shadow('x',   3, alpha=0.2)
line.add_shadow('z', -25, alpha=0.2)

plt  = Plotter(axes=dict(xygrid=False))
plt.show(__doc__, line, viewup='z').close()
