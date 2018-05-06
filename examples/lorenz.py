from __future__ import division, print_function
from plotter import vtkPlotter, mag
import numpy as np

scene = vtkPlotter(title='Lorenz attractor', axes=2)
scene.verbose = 0

dt = 0.001
y = [25, -10, -7] # Starting point (initial condition)
pts, cols = [], []
scene.point(y, r=20, c='g', alpha=0.3)

for t in np.linspace(0,20, int(20/dt)):
  # Integrate a funny differential equation
  dydt = np.array([-8/3*y[0]+ y[1]*y[2], -10*(y[1]-y[2]), -y[1]*y[0]+28*y[1]-y[2]])
  y = y + dydt * dt

  c = np.clip( [mag(dydt) * 0.005], 0, 1)[0] # color by speed
  pts.append(y)
  cols.append([c,0, 1-c])

scene.points(pts, cols, r=2)
scene.show()

