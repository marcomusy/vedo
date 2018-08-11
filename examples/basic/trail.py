# Example use of addTrail() and updateTrail()
#
from vtkplotter import Plotter, sin

vp = Plotter(axes=1, interactive=0)

c = vp.cube()
vp.cutPlane(c, [-0.4,0,0]) #cut away the left face

s = vp.sphere([1,1,1], r=.03, c='db')

# add a trail to last created actor with max 50 segments
vp.addTrail(c='k', lw=3, maxlength=.5, n=50) 

for i in range(200):
    s.pos([-2. +i/100., sin(i/5.)/10., 0]).updateTrail()
    vp.render()
    vp.camera.Azimuth(-.3)

vp.show(interactive=1)