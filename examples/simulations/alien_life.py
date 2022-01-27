# Copyright (c) 2017, Nicolas P. Rougier https://github.com/rougier/alien-life
# Adapted for vedo by M. Musy, June 2021
#
import numpy as np
from noise._simplex import noise4
from vedo import Plotter, Points, Circle, Text2D

n = 50000
radius = 200
width, height = (500, 500)
scale = 0.005
t = 0

T = np.random.uniform(0, 2*np.pi, n)
R = np.sqrt(np.random.uniform(0, 1, n))
P = np.zeros((n,2))
X,Y = P[:,0], P[:,1]
X[...] = R*np.cos(T)
Y[...] = R*np.sin(T)
intensity = np.power(1.001-np.sqrt(X**2 + Y**2), 0.75)
X[...] = X*radius + width//2
Y[...] = Y*radius + height//2


def update(evt):
    global t
    t += 0.005
    P_ = np.zeros((n,3))
    cos_t = 1.5*np.cos(2*np.pi * t)
    sin_t = 1.5*np.sin(2*np.pi * t)
    for i in range(n):
        x, y = P[i]
        f = intensity[i]*50
        dx = noise4(scale*x, scale*y, cos_t, sin_t, 2) * f
        dy = noise4(100+scale*x, 200+scale*y, cos_t, sin_t, 2) * f
        P_[i] = [x + dx, y + dy, np.sqrt(dx*dx+dy*dy)/2]
    pts.points(P_)
    plt.render()


pts = Points([X, Y], r=3).alpha(0.8)
cir = Circle(pos=(width/2, height/2, -5), r=radius*1.05)
txt1= Text2D("\Lambda  L  I  E  N    L  I  F  E", s=2.8, pos="top-center")
txt2= Text2D("Original idea by Necessary Disorder", s=0.9, pos="bottom-center")

plt = Plotter()
plt.show(pts, cir, txt1, txt2, elevation=-35, zoom=1.2, interactive=False)
plt.addCallback("timer", update)
plt.timerCallback("create")
plt.interactive()
