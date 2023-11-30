"""Visualize scalar values interactively
by hovering the mouse on a mesh
Press c to clear the path"""
from vedo import *

def func(evt):                       ### called every time mouse moves!
    msh = evt.object                 # get the mesh that triggered the event
    if not msh:
        return                       # mouse hits nothing, return.
    pt  = evt.picked3d               # 3d coords of point under mouse
    pid = msh.closest_point(pt, return_point_id=True)
    txt =(
        f"Point:  {precision(pt[:2]  ,2)}\n"
        f"Height: {precision(arr[pid],3)}\n"
        f"Ground speed: {precision(evt.speed3d*100,2)}"
    )
    msg.text(txt)                    # update text message

    ar = Arrow(pt - evt.delta3d, pt, s=0.001, c='orange5')
    fp = msh.flagpole(
        txt, point=pt,s=0.04, c='k', font="VictorMono",
    )
    fp.follow_camera()                 # make it always face the camera
    plt.remove("FlagPole").add(ar, fp) # remove the old flagpole, add the new
    plt.render()

msg = Text2D(pos='bottom-left', font="VictorMono") # an empty text
hil = ParametricShape('RandomHills').cmap('terrain').add_scalarbar()
arr = hil.pointdata["Scalars"]       # numpy array with heights

plt = Plotter(axes=1, bg2='lightblue')
plt.add_callback('mouse move', func) # add the callback function
plt.add_callback('keyboard', lambda _: plt.remove("Arrow").render())
plt.show(hil, msg, __doc__, viewup='z')
plt.close()

