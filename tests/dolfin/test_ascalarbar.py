import numpy as np
from dolfin import *
from dolfin import __version__
from vedo.dolfin import plot, screenshot, MeshActor, show
from vedo import settings


print('Test ascalarbar, dolfin version', __version__)

if hasattr(MPI, 'comm_world'):
    mesh = UnitSquareMesh(MPI.comm_world, nx=16, ny=16)
else:
    mesh = UnitSquareMesh(16,16)

V = FunctionSpace(mesh, 'Lagrange', 1)
f = Expression('10*(x[0]+x[1]-1)', degree=1)
u = interpolate(f, V)

actors = plot(u, mode='color', cmap='viridis', vmin=-3, vmax=3, style=1,
              returnActorsNoShow=True)

actor = actors[0]

solution = actor.pointdata[0]

print('ArrayNames', actor.pointdata.keys())
print('min', 'mean', 'max:')
print(np.min(solution), np.mean(solution), np.max(solution), len(solution))

assert len(solution) == 289
assert np.isclose(np.min(solution) , -10., atol=1e-05)
assert np.isclose(np.max(solution) ,  10., atol=1e-05)


