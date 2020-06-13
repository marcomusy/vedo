"""Usage example of fitLine() and fitPlane()

Draw a line in 3D that fits a cloud of 20 Points,
Show the first set of 20 points and fit a plane to them.
"""
import numpy as np
from vedo import *

# declare the class instance
vp = Plotter()

# draw 500 fit lines superimposed and very transparent
for i in range(500):

    x = np.linspace(-2, 5, 20)  # generate every time 20 points
    y = np.linspace(1, 9, 20)
    z = np.linspace(-5, 3, 20)
    data = np.stack((x,y,z), axis=1)
    data += np.random.normal(size=data.shape) * 0.8  # add gauss noise

    vp += fitLine(data).lw(4).alpha(0.04)  # fit a line

# 'data' still contains the last iteration points
vp += Points(data, r=10, c="yellow")

print("Line 0 Fit slope = ", vp.actors[0].slope)

plane = fitPlane(data)  # fit a plane
print("Plane Fit normal =", plane.normal)

vp += [plane, __doc__]

vp.show(axes=1)
