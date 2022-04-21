"""The Lorenz attractor is a set of chaotic solutions of
a particular system of ordinary differential equations"""
import numpy as np

dt = 0.002
y = (25.0, -10.0, -7.0)  # Starting point (initial condition)
pts, cols = [], []

for t in np.linspace(0, 20, int(20 / dt)):
    # Integrate a funny differential equation
    dydt = np.array(
        [-8 / 3.0 * y[0] + y[1] * y[2],
         -10.0 * (y[1] - y[2]),
         -y[1] * y[0] + 28.0 * y[1] - y[2]]
    )
    y = y + dydt * dt
    c = np.clip([np.linalg.norm(dydt) * 0.005], 0, 1)[0]  # color by speed
    cols.append([c, 0, 1-c])
    pts.append(y)


from vedo import *

scene  = Plotter(title="Lorenz attractor", axes=dict(yzGrid=True))
scene += Point(y, r=10, c="g") # end point
scene += Points(pts, r=3, c=cols)
scene += Line(pts).off().addShadow(plane='x', point=3) # only show shadow, not line
scene += Line(pts).off().addShadow(plane='z', point=-30)
scene += __doc__
scene.show(viewup='z').close()
