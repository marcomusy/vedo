# Plot a volume evolution in time
# Credits: https://github.com/edmontz
import numpy as np
from scipy.fftpack import fftn, fftshift
from vedo import Axes, Plotter, Volume, progressbar


def f(x, y, z, t):
    r = np.sqrt(x*x + y*y + z*z + 2*t*t) + 0.1
    return np.sin(9*np.pi * r)/r

n = 64
qn = 25
vol = np.zeros((n, n, n))
n1 = int(n/2)

plt = Plotter(bg="black", interactive=False)
axes = Axes(xrange=(0,n), yrange=(0,n), zrange=(0,n))
plt.show(axes, viewup='z')

for q in progressbar(range(qn), c='r'):
    t = 2 * q / qn - 1
    for k in range(n1):
        z = 2 * k / n1 - 1
        for j in range(n1):
            y = 2 * j / n1 - 1
            for i in range(n1):
                x = 2 * i / n1 - 1
                vol[i, j, k] = f(x, y, z, t)
    volf = fftn(vol)
    volf = fftshift(abs(volf))
    volf = np.log(12*volf/volf.max()+ 1) / 2.5

    volb = Volume(volf)
    volb.mode(1).cmap("rainbow").alpha([0, 0.8, 1])
    volb.name = "MyVolume"
    plt.remove("MyVolume").add(volb).render()

plt.interactive().close()
