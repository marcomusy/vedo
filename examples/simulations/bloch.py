"""Animate text and meshes"""
from vedo import *

v = Arrow([0,0,0],[0,.71,.71]).addTrail(maxlength=0.1, c='k').addShadow(z=-1)
s = Sphere().c('red5', 0.1)

cam = dict(pos=(4.14, -4.25, 2.35),
           focalPoint=(0.167, -0.287, 0.400),
           viewup=(-0.230, 0.235, 0.944),
           distance=5.94)
# (press "C" in rendering window to get the above camera settings)

msg1 = Text2D("\n The Bloch sphere \n",
              pos="top-center", font=2, c='w', bg='b3', alpha=1)
msg2 = Text3D("|\Psi> state", font=10, c='k', italic=1).scale(.08).followCamera()
axs = Axes(xrange=(-1,1), yrange=(-1,1), zrange=(-1,2), yzGrid=False)

plt = show(s, v, msg1, msg2, axs, camera=cam, interactive=False)

# vd = Video()
for i in range(200):
    v.rotateZ(3.6)
    msg2.pos(v.centerOfMass()+[0.2,0,0])
    plt.show(s, v, msg2)
    # vd.addFrame()
# vd.close()
interactive().close()
