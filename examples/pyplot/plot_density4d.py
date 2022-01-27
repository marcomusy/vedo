# Plot a volume evolution in time
# Credits: https://github.com/edmontz
import numpy as np
from scipy.fftpack import fftn, fftshift
from vedo import Volume, ProgressBar, show, settings

settings.allowInteraction = True

def f(x, y, z, t):
    r = np.sqrt(x*x + y*y + z*z + 2*t*t) + 0.1
    return np.sin(9*np.pi * r)/r

n = 64
qn = 50
vol = np.zeros((n, n, n))
n1 = int(n/2)

pb = ProgressBar(0, qn, c="r")
for q in pb.range():
    pb.print()

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

    vb = Volume(volf).mode(1).c("rainbow").alpha([0, 0.8, 1])
    plt = show(vb, bg="black", axes=1, viewup='z', interactive=False)
    if plt.escaped: break  # ESC button was hit

plt.interactive().close()
