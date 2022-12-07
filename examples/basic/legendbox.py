"""Customizing a legend box"""
from vedo import *

s = Sphere()
c = Cube().x(2)
e = Ellipsoid().x(4)
h = Hyperboloid().x(6).legend('The description for\nthis one is quite long')

lb = LegendBox([s,c,e,h], width=0.3, height=0.4, markers='s').font("Kanopus")

cam = dict( # press C in window to get these numbers
    position=(10.4414, -7.62994, 4.18818),
    focal_point=(4.10196, 0.335224, -0.148651),
    viewup=(-0.252830, 0.299657, 0.919936),
    distance=11.0653,
    clipping_range=(3.69605, 21.2641),
)

show(s, c, e, h, lb, __doc__,
     axes=1, bg='lightyellow', bg2='white', size=(1400,800), camera=cam
).close()

