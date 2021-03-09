"""Slice a Volume with an arbitrary plane
hover the plane to get the scalar values"""
from vedo import *

vol = Volume(dataurl+'embryo.slc').alpha([0,0,0.8]).c('w').pickable(False)

sl = vol.slicePlane(origin=vol.center(), normal=(0,1,1))
sl.cmap('Purples_r').lighting('off').addScalarBar(title='Slice', c='w')
arr = sl.getPointArray()

def func(evt):
    if not evt.actor: return

    ptid= evt.actor.closestPoint(evt.picked3d, returnPointId=True)
    txt = f"Probing:\n{precision(evt.actor.picked3d, 3)}\nvalue = {arr[ptid]}"
    sph = Sphere(evt.actor.points(ptid), c='orange7').pickable(False)
    vig = sph.vignette(txt, s=7, offset=(-150,15), font=2)#.followCamera()
    msg = Text2D(txt, pos='bottom-left', font="VictorMono")
    plt.remove(plt.actors[-3:])   # remove the last 3
    plt.add([sph, vig, msg])      # add the new 3 ones

plt = show(vol, sl, __doc__, axes=9, bg='k', bg2='bb', interactive=False)
plt.actors += [None, None, None]  # holds [sphere, vignette, text2d]
plt.addCallback('MouseMove', func)
interactive()
