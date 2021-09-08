"""Slice a Volume with an arbitrary plane
hover the plane to get the scalar values"""
from vedo import *

vol = Volume(dataurl+'embryo.slc').alpha([0,0,0.8]).c('w').pickable(False)

sl = vol.slicePlane(origin=vol.center(), normal=(0,1,1))
sl.cmap('Purples_r').lighting('off').addScalarBar(title='Slice', c='w')
arr = sl.pointdata[0] # retrieve vertex array data

def func(evt):
    if not evt.actor:
        return
    pid = evt.actor.closestPoint(evt.picked3d, returnPointId=True)
    txt = f"Probing:\n{precision(evt.actor.picked3d, 3)}\nvalue = {arr[pid]}"
    sph = Sphere(evt.actor.points(pid), c='orange7').pickable(False)
    vig = sph.vignette(txt, s=7, offset=(-150,15), font=2).followCamera()
    plt.remove(plt.actors[-2:]).add([sph, vig]) #remove old 2 & add the new 2

plt = show(vol, sl, __doc__, axes=9, bg='k', bg2='bb', interactive=False)
plt.actors += [None, None]  # 2 placeholders for [sphere, vignette]
plt.addCallback('as my mouse moves please call', func) # be kind to vedo
interactive().close()
