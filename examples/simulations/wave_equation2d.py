"""Solve the wave equation using finite differences and Forward Euler"""
# Original code based on:
# Based on: https://commons.wikimedia.org/wiki/File:Heat_eqn.gif
# Adapted for vedo by M. Musy, 20022
import numpy as np
from scipy.ndimage import gaussian_filter
from vedo import Plotter, settings, Grid, show


def make_step(Z0, Z):
    ntime_anim = int(time.shape[0] / nframes)
    for cont in range(ntime_anim):
        Z_aux = Z.copy()
        Z[1:N-1, 1:N-1] = (
            2 * Z[1:N-1, 1:N-1]
            - Z0[1:N-1, 1:N-1]
            + (dt / dx) ** 2
            * (
                Z[2:N, 1:N-1]
                + Z[0:N - 2, 1:N-1]
                + Z[1:N-1, 0:N - 2]
                + Z[1:N-1, 2:N]
                - 4 * Z[1:N-1, 1:N-1]
            )
        )
        Z0[:] = Z_aux[:]
    return Z0, Z


N = 500      # Grid resolution
A, B = 5, 4  # Box sizes
end_time = 5
nframes = 150

X, Y = np.mgrid[-A:A:N*1j, -B:B:N*1j]
dx = X[1, 0] - X[0, 0]
dt = 0.1 * dx
time = np.arange(0, end_time, dt)

# initial condition
Z0 = np.ones_like(X)
Z0[X ** 2 + Y ** 2 < 0.5] = 0
Z0[X ** 2 + Y ** 2 > 2] = 0
Z0 = gaussian_filter(Z0, sigma=2)
Z1 = np.array(Z0)

# plot stuff
grid = Grid(sx=X[:,0], sy=Y[0], lw=0).lighting('glossy')

cam = dict(pos=(5.715, -10.54, 12.72),
           focalPoint=(0.1380, -0.7437, -0.5408),
           viewup=(-0.2242, 0.7363, 0.6384),
           distance=17.40,
)

settings.allowInteraction = True
settings.defaultFont = "Ubuntu"

plotter = show(grid, __doc__,
                axes=1, camera=cam,
                size=(1000,700), interactive=False,
)

for cont in range(nframes):
    Z0, Z1 = make_step(Z0, Z1)
    wave = Z1.ravel()
    # print(wave.min(), wave.max())
    grid.cmap("Blues", wave, vmin=-1.5, vmax=1.5)
    newpts = grid.points()
    newpts[:,2] = wave
    grid.points(newpts)
    plotter.render()#show(resetcam=0)

plotter.interactive().close()
