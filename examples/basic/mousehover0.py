"""Use a flagpost object to visualize some property interactively"""
from vedo import ParametricShape, Plotter, precision

def func(evt):  ### called every time mouse moves!
    if not evt.object:
        return  # mouse hits nothing, return.

    pt = evt.picked3d  # 3d coords of point under mouse
    txt = f"Position: {precision(pt[:2],2)}\n" \
          f"Speed   : {precision(evt.speed3d*100,2)} km/h"
    flagpost.text(txt).pos(pt)  # update text and position
    plt.render()

hil = ParametricShape('RandomHills').cmap('terrain')
flagpost = hil.flagpost(offset=(0,0,0.25))

plt = Plotter(axes=1, bg2='yellow9', size=(1150, 750))
plt.add_callback('mouse move', func) # add the callback function
plt.show(hil, flagpost, __doc__, viewup='z', zoom=2)
plt.close()
