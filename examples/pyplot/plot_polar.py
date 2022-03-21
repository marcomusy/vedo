"A splined polar plot"
from vedo import *
from vedo.pyplot import plot


angles    = vector([  0,  20,  60, 160, 200, 250, 300, 340])
distances = vector([0.1, 0.2, 0.3, 0.5, 0.6, 0.4, 0.2, 0.1])

dn1 = plot(angles, distances,
           mode='polar', deg=True, splined=True, fill=True,
           c='green', bc='k', alpha=0.7, title=__doc__, vmax=0.65)

dn2 = plot(angles+120, distances**2,
           mode='polar', deg=True, splined=True, fill=True,
           c='red', alpha=1, vmax=0.65)
dn2.z(0.01) # set a positive z so it stays in front

show(dn1, dn2, zoom=1.2, bg='k9').close()
