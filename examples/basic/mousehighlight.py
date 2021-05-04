"""Click a sphere to highlight it"""
from vedo import Text2D, Sphere, Plotter
import numpy as np

spheres = []
for i in range(25):
    p = np.random.rand(2)
    s = Sphere(r=0.05).pos(p).color('k5')
    s.name = f"sphere nr.{i} at {p}"
    spheres.append(s)

def func(evt):
    if not evt.actor: return
    sil = evt.actor.silhouette().lineWidth(6).c('red5')
    msg.text("You clicked: "+evt.actor.name)
    plt.remove(silcont.pop()).add(sil)
    silcont.append(sil)

silcont = [None]
msg = Text2D("", pos="bottom-center", c='k', bg='r9', alpha=0.8)

plt = Plotter(axes=1, bg='black')
plt.addCallback('mouse click', func)
plt.show(spheres, msg, __doc__, zoom=1.2).close()
