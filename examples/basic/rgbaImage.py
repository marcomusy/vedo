"""example plot of 2 images containing an
alpha channel for modulating the opacity
"""
from vedo import Plotter, Picture
import numpy as np

vp = Plotter(axes=3)

rgbaimage = np.random.rand(50,50,4)*255
alpharamp = np.linspace(0,255,50).astype(int)
rgbaimage[:,:,3] = alpharamp
rgbaimage2 = np.random.rand(50,50,4)*255
rgbaimage2[:,:,3] = alpharamp[::-1]

p1 = Picture(rgbaimage)
vp += p1

p2 = Picture(rgbaimage2)
p2.pos(0,0,5)
vp += p2

vp += __doc__
vp.show()
