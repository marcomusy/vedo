"""Generate the Mandelbrot set as a color-mapped Image object"""
import numpy as np
from vedo import Image, dataurl, show


def mandelbrot(h=400, w=400, maxit=20, r=2):
    # Returns an image of the Mandelbrot fractal of size (h,w)
    x = np.linspace(-2.5, 1.5, 4*h+1)
    y = np.linspace(-1.5, 1.5, 3*w+1)
    A, B = np.meshgrid(x, y)
    C = A + B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)
    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r                    # who is diverging
        div_now = diverge & (divtime == maxit)  # who is diverging now
        divtime[div_now] = i                    # note when
        z[diverge] = r                          # avoid diverging too much
    return divtime

img = Image(mandelbrot()).cmap("RdGy")
show(img, __doc__, axes=1, size=[800,600], zoom=1.4).close()
