"""Interpolate gap between two functions"""
# https://www.youtube.com/watch?v=vD5g8aVscUI
import numpy as np
from vedo.pyplot import plot
from vedo import settings

settings.remember_last_figure_format = True # useful for pf += plot(...)

x1 = np.linspace(-2,2, num=100)
x  = np.linspace(-2,2, num=100)
x2 = np.linspace(-2,2, num=100)

fx = np.sin(x1*3) - 1
gx = x2*x2/3 -1

def phi(x):
    psi = np.exp(-1/x)
    psi_1 = np.exp(-1/(1-x))
    phi = psi / (psi + psi_1)
    phi = np.where(x<=0, 0, phi)
    phi = np.where(x>1, 1, phi)
    return phi

w = phi(x)
h = (1-w) * fx + w * gx

pf  = plot(x1[:50], fx[:50], xlim=[-2,2], ylim=[-2,1.5], lw=5, title=__doc__)
pf += plot(x[50:75], h[50:75], c='red5')
pf += plot(x2[75:], gx[75:], lw=5)
pf += plot(x[50:75], w[50:75], c='green4', lw=1)

pf.show(mode='image', zoom='tight')

