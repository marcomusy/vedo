"""Plot streamlines of the 2D field:

u(x,y) = -1 - x\^2 + y
v(x,y) =  1 + x  - y\^2
"""
from vedo import Points, show
from vedo.pyplot import streamplot
import numpy as np

# a grid with a vector field (U,V):
X, Y = np.mgrid[-5:5 :15j, -4:4 :15j]
U = -1 - X**2 + Y
V =  1 + X    - Y**2

# optionally, pick some random points as seeds:
prob_pts = np.random.rand(200, 2)*8 - [4,4]

sp = streamplot(
    X,Y, U,V,
    lw=0.001,            # line width in abs. units
    direction='forward', # 'both' or 'backward'
    probes=prob_pts,
)

pts = Points(prob_pts, r=5, c='white')

show(sp, pts, __doc__, axes=1, bg='bb').close()
