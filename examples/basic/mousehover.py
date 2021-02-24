"""Visualize scalar values interactively
by hovering the mouse on a mesh
Press c to clear"""
from vedo import *

def func(evt):                     ### called every time the mouse moves!
    if not evt.actor: return       # no hits, return. (NB: evt is a dictionary)
    pt = evt.picked3d              # 3d coords of picked point under mouse

    pid = evt.actor.closestPoint(pt, returnPointId=True)
    txt = f"Point:  {precision(pt[:2],2)}\n"  \
          f"Height: {precision(arr[pid],3)}\n"\
          f"Ground speed: {precision(evt.speed3d*100,2)}"
    arw = Arrow(pt - evt.delta3d, pt, s=0.001, c='orange5')
    vig = evt.actor.vignette(txt, point=pt, offset=(0.4,0.6),
                             s=0.04, c='k', font="VictorMono").followCamera()
    msg = Text2D(txt, pos='bottom-left', font="VictorMono")
    if len(plt.actors) > 3:
        plt.remove(plt.actors[-2:]) # remove the old vig and msg
    plt.add([arw, vig, msg])        # add Arrow and the new vig and msg

hil = ParametricShape('RandomHills').cmap('terrain').addScalarBar()
arr = hil.getPointArray("Scalars")

plt = Plotter(axes=1, bg2='lightblue')
plt.addCallback('MouseMove', func)  # the callback function
plt.addCallback('KeyPress', lambda e: plt.remove(plt.actors[2:], render=True))
plt.show(hil, __doc__, viewup='z')
