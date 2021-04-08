"""Customizing a legend box"""
from vedo import *

s = Sphere()
c = Cube().x(2)
e = Ellipsoid().x(4)
h = Hyperboloid().x(6).legend('The description for\nthis one is quite long')

lb = LegendBox([s,c,e,h], width=0.3, height=0.4).font(5)

show(s, c, e, h, lb, __doc__,
     axes=1, bg='ly', bg2='w', size=(1400,800), viewup='z')

