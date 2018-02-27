import plotter
import numpy as np

scene = plotter.vtkPlotter(title='Lorenz differential equation', axes=0)
scene.verbose = 0

dt = 0.001
y = [25., -10., -7.] # Starting point (initial condition)
lorenz, cols = [], []
scene.grid(pos=y, s=50)

for t in np.linspace(0,20, int(20/dt)):
  # Integrate a funny differential equation
  dydt = np.array([-8./3*y[0]+ y[1]*y[2], -10*(y[1]-y[2]), -y[1]*y[0]+28*y[1]-y[2]])
  y = y + dydt * dt

  c = np.clip( [np.linalg.norm(dydt) * 0.005], 0, 1)[0] # color by speed
  lorenz.append( y )
  cols.append((c,0, 1-c))

scene.points(lorenz, cols)
scene.show()

