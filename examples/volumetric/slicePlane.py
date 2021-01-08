"""Slice a Volume with an arbitrary plane
click the plane to get the scalar value"""
from vedo import *

vol = Volume(datadir+'embryo.slc').alpha([0,0,0.5]).c('hot_r')
vol.pickable(False).addScalarBar3D(title='Voxel intensity', c='k')

sl = vol.slicePlane(origin=vol.center(), normal=(0,1,1))
sl.cmap('viridis').lighting('ambient').addScalarBar(title='Slice')

plt = show(vol, sl, __doc__, axes=3, interactive=False)

def func(mesh):
    ptid = mesh.closestPoint(mesh.picked3d, returnPointId=True)
    val = arr[ptid]
    txt = precision(mesh.picked3d, 3) + f"\nvalue = {val}"
    vpt = Sphere(mesh.points(ptid), r=1, c='pink').pickable(0)
    vig = vpt.vignette(txt, s=5, offset=(20,10)).followCamera()
    msg = Text2D(txt, pos='bottom-left', font="VictorMono")
    plt.remove(plt.actors[-3:], render=False) # remove the last 3
    plt.add([vpt, vig, msg])                  # add the new ones

arr = sl.getPointArray()
plt.actors += [None,None,None] # holds sphere, vignette, text2d
plt.mouseLeftClickFunction = func
plt.resetcam = False
interactive()