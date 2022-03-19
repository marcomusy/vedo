"""Forward kinematics: hover the mouse to drag the chain"""
from vedo import Plotter, versor, Plane, Line

n = 15  # number of points
l =  3  # length of one segment

def func(evt):
    if not evt.actor:
        return
    coords = line.points()
    coords[0] = evt.picked3d
    for i in range(1, n):
        v = versor(coords[i] - coords[i-1])
        coords[i] = coords[i-1] + v * l
    line.points(coords)  # update positions
    nodes.points(coords)
    plt.render()

surf = Plane(s=[60, 60])
line = Line([l*n/2, 0], [-l*n/2, 0], res=n, lw=12)
nodes= line.clone().c('red3').pointSize(15)

plt = Plotter()
plt.addCallback("on mouse move please call", func)
plt.show(surf, line, nodes, __doc__, zoom=1.3)
plt.close()
