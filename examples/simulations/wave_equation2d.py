"""Solve the wave equation using finite differences and the Euler method
(press space bar to continue)"""
import numpy as np
from scipy.ndimage import gaussian_filter
from vedo import Grid, Text2D, show

N = 400      # grid resolution
A, B = 5, 4  # box sides
end = 5      # end time
nframes = 150

X, Y = np.mgrid[-A:A:N*1j, -B:B:N*1j]
dx = X[1,0] - X[0,0]
dt = 0.1 * dx
time = np.arange(0, end, dt)
m = int(len(time)/nframes)

# initial condition (a ring-like wave)
Z0 = np.ones_like(X)
Z0[X**2+Y**2 < 1] = 0
Z0[X**2+Y**2 > 2] = 0
Z0 = gaussian_filter(Z0, sigma=4)
Z1 = np.array(Z0)

grid = Grid(s=(X[:,0], Y[0])).lineWidth(0).lighting('glossy')
txt = Text2D(font='Brachium', pos='bottom-left', bg='yellow5')

cam = dict(
    pos=(5.715, -10.54, 12.72),
    focalPoint=(0.1380, -0.7437, -0.5408),
    viewup=(-0.2242, 0.7363, 0.6384),
    distance=17.40,
)
plt = show(grid, txt, __doc__,
           camera=cam, axes=1, size=(1000,700), interactive=False,
)

for i in range(nframes):
    # iterate m times before showing the frame
    for _ in range(m):
        ZC = Z1.copy()
        Z1[1:N-1, 1:N-1] = (
            2*Z1[1:N-1, 1:N-1]
            - Z0[1:N-1, 1:N-1]
            + (dt/dx)**2
            * (  Z1[2:N,   1:N-1]
               + Z1[0:N-2, 1:N-1]
               + Z1[1:N-1, 0:N-2]
               + Z1[1:N-1, 2:N  ]
               - 4*Z1[1:N-1, 1:N-1] )
        )
        Z0[:] = ZC[:]

    wave = Z1.ravel()
    txt.text(f"frame: {i}/{nframes}, higth_max = {wave.max()}")
    grid.cmap("Blues", wave, vmin=-2, vmax=2)
    newpts = grid.points()
    newpts[:,2] = wave
    grid.points(newpts)  # update the z component
    plt.render().interactive()

plt.close()
