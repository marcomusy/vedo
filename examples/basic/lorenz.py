from vtkplotter import Plotter, mag, Points, Point
import numpy as np


dt = 0.002
y = [25.0, -10.0, -7.0]  # Starting point (initial condition)
pts, cols = [], []


for t in np.linspace(0, 20, int(20 / dt)):
    # Integrate a funny differential equation
    dydt = np.array(
        [-8 / 3.0 * y[0] + y[1] * y[2], -10.0 * (y[1] - y[2]), -y[1] * y[0] + 28.0 * y[1] - y[2]]
    )
    y = y + dydt * dt

    c = np.clip([mag(dydt) * 0.005], 0, 1)[0]  # color by speed
    pts.append(y)
    cols.append([c, 0, 1 - c])

scene = Plotter(title="Lorenz attractor", axes=2, verbose=0, bg="w")
scene += Point(y, r=20, c="g", alpha=0.3)
scene += Points(pts, r=5, c=cols, alpha=0.3)
scene.show()
