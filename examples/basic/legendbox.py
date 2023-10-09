"""Customizing a legend box"""
from vedo import *

s = Sphere()
c = Cube().x(2)
e = Ellipsoid().x(4)

h = Hyperboloid().x(6)
h.legend('The description for\nthis one is quite long')

lbox = LegendBox([s,c,e,h], width=0.3, height=0.4, markers='s')
lbox.font("Kanopus")

show(s, c, e, h, lbox, __doc__,
     axes=1, bg='lightyellow', bg2='white', size=(1200,800), viewup='z'
).close()

