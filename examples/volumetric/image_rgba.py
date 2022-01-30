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

# Second example: a b&w image from a numpy array
img = np.zeros([512,512])
img[0:256, 0:256] =   0
img[0:256,  256:] =  64
img[256:,  0:256] = 128
img[256:,   256:] = 255
img = img.transpose(1,0)

pict = Picture(img)

show(pict, mode="image", bg=(0.4,0.5,0.6), axes=1).close()
