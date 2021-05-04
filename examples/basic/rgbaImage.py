"""Example plot of 2 images containing an
alpha channel for modulating the opacity"""
#Credits: https://github.com/ilorevilo
from vedo import Picture, show
import numpy as np

rgbaimage1 = np.random.rand(50, 50, 4) * 255
alpharamp = np.linspace(0, 255, 50).astype(int)
rgbaimage1[:, :, 3] = alpharamp
rgbaimage2 = np.random.rand(50, 50, 4) * 255
rgbaimage2[:, :, 3] = alpharamp[::-1]

p1 = Picture(rgbaimage1)

p2 = Picture(rgbaimage2).z(12)

show(p1, p2, __doc__, axes=7, viewup="z").close()
