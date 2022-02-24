"""Find the catenary line connecting two points
(hover mouse)"""
# Credits:
# https://trinket.io/glowscript/f938182b85
# https://www.youtube.com/watch?v=gkdnS6rISV0
import numpy as np
import vedo

P1 = (0.0, 0.0)
P2 = (0.0, 0.0)
L = 1.6
n = 15
dA = 0.001

def move(evt):
    if not evt.actor:
        return
    x0, y0 = P1
    x1, y1, _ = evt.picked3d
    dx = x1 - x0
    xb = (x1 + x0) / 2
    dy = y1 - y0
    r = np.sqrt(L**2 - dy**2) / dx

    A = 0.01
    left = r * A
    right = np.sinh(A)
    while left >= right:
        left = r * A
        right = np.sinh(A)
        A = A + dA
    A = A - dA

    a = dx / (2 * A)
    b = xb - a * 1/2 * np.log((1+dy/L)/(1-dy/L))
    c = y0 - a * np.cosh((x0-b)/a)
    x = x0
    ddx = 0.001
    pts = []
    while x < x1:
        y = a * np.cosh((x-b)/a) + c
        x = x + ddx
        pts.append([x, y])
    pts.append([x1, y1])

    coords = vedo.Spline(pts, res=n).points()
    line.points(coords)  # update coords
    nodes.points(coords).z(.01)
    plt.render()

surf = vedo.Plane().shift(0.51,0,0)
line = vedo.Line(P1, P2, res=n, lw=10)
nodes= line.clone().c('red3').pointSize(8).z(.01)

plt = vedo.Plotter()
plt.addCallback("hover", move)
plt.show(__doc__, surf, line, nodes, axes=1, mode='image')

