"""Click a sphere to highlight it"""
from vedo import Sphere, Plotter
import numpy as np

pts = np.random.rand(30,2)*20
spheres = [Sphere().pos(p).color('k5') for p in pts]

def func(evt):
    if not evt.actor: return
    sil = evt.actor.silhouette().lineWidth(6).c('red5')
    plt.remove(silcont.pop()).add(sil)
    silcont.append(sil)

silcont = [None]
plt = Plotter(axes=1, bg='black')
plt.addCallback('mouse click', func)
plt.show(spheres, __doc__, zoom=1.2)
