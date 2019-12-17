import numpy as np
from dolfin import *
from vtkplotter.dolfin import plot, screenshot
from vtkplotter import settings


print('Test ascalarbar')

mesh = UnitSquareMesh(MPI.comm_world, nx=16, ny=16)
V = FunctionSpace(mesh, 'Lagrange', 1)
f = Expression('10*(x[0]+x[1]-1)', degree=1)
u = interpolate(f, V)
plot(u, mode='color', cmap='viridis', vmin=-3, vmax=3, style=1, offscreen=1)

screenshot('ascalarbar.png')

actor = settings.plotter_instance.actors[0]
solution = actor.scalars(0)

print('ArrayNames', actor.getArrayNames())
print('min', 'mean', 'max:')
print(np.min(solution), np.mean(solution), np.max(solution), len(solution))

assert len(solution) == 289
assert np.isclose(np.min(solution) , -10., atol=1e-05)
assert np.isclose(np.max(solution) ,  10., atol=1e-05)


