"""Customizing a legend box"""
from vedo import *

s = Sphere()
c = Cube().x(2)
e = Ellipsoid().x(4)
h = Hyperboloid().x(6).legend('The description for\nthis one is quite long')

lb = LegendBox([s,c,e,h], width=0.3, height=0.4).font(5)

cam = dict(pos=(10.1, -8.33, 7.25),  # params obtained by pressing "C"
           focalPoint=(4.46, 1.31, -0.644),
           viewup=(-0.379, 0.443, 0.813),
           distance=13.7,
)

show(s, c, e, h, lb, __doc__,
     axes=1, bg='lightyellow', bg2='white', size=(1400,800), camera=cam,
).close()

