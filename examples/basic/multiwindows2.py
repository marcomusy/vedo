"""Multiple plotter sync-ed windows"""
from vedo import *

acts = [Ellipsoid().color('Bisque'),
        Cone().color('RosyBrown'),
        Cylinder().color('Chocolate'),
]

opts = dict(axes=1, interactive=False, new=True, size=(390,390))
ts = [f"Window nr.{i}" for i in range(4)]

plt0 = show(acts[0], **opts, pos=( 200,0), title=ts[0], viewup='z')
plt1 = show(acts[1], **opts, pos=( 600,0), title=ts[1], camera=plt0.camera)
plt2 = show(acts[2], __doc__, **opts, pos=(1000,0), title=ts[2], camera=plt0.camera)
plts = [plt0, plt1, plt2]

def func(evt):
    for i in range(3):
        if ts[i] != evt.title: # only update the other windows
            plts[i].render()

for plt in plts:
    plt.addCallback('Interaction', func)
    plt.addCallback('EndInteraction', func) # because zooming is not an "Interaction" event

interactive()
