"""Visualize scalar values interactively
by hovering the mouse on a mesh
Press c to clear the path"""
from vedo import *

def func(evt):                       ### called every time mouse moves!
    msh = evt.actor
    if not msh:
        return                       # mouse hits nothing, return.
    pt = evt.picked3d                # 3d coords of point under mouse

    pid = msh.closestPoint(pt, returnPointId=True)
    txt = f"Point:  {precision(pt[:2]  ,2)}\n" \
          f"Height: {precision(arr[pid],3)}\n" \
          f"Ground speed: {precision(evt.speed3d*100,2)}"
    msg.text(txt)                    # update text message
    arw = Arrow(pt - evt.delta3d, pt, s=0.001, c='orange5')
    vig = msh.vignette(txt, point=pt, offset=(0.4,0.6),
                       s=0.04, c='k', font="VictorMono")
    vig.followCamera()               # make it always face the camera
    if len(plt.actors) > 3:
        plt.pop()                    # remove the old vignette
    plt.add(arw, vig)                # add Arrow and the new vignette

msg = Text2D(pos='bottom-left', font="VictorMono") # an empty text
hil = ParametricShape('RandomHills').cmap('terrain').addScalarBar()
arr = hil.pointdata["Scalars"]       # numpy array with heights

plt = Plotter(axes=1, bg2='lightblue')
plt.addCallback('mouse move', func)  # add the callback function
plt.addCallback('keyboard', lambda evt:
                    plt.remove(plt.actors[3:]).render()
)
plt.show(hil, msg, __doc__, viewup='z')
plt.close()

