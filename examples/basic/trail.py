# Example use of addTrail() and updateTrail()
#
from vtkplotter import Plotter, sin, Assembly

vp = Plotter(axes=6)

c = vp.sphere(c='green', res=24)
vp.cutPlane(c, [-0.9,0,0], showcut=True) #cut left part of sphere

s = vp.sphere([1,1,1], r=.03, c='db')

# add a trail to last created actor with max 50 segments
vp.addTrail(c='k', lw=3, maxlength=.5, n=50) 

for i in range(200):
    s.pos([-2 +i/100., sin(i/5.)/10, 0]).updateTrail()
    vp.render()
    vp.camera.Azimuth(-0.2)

vp.show(resetcam=0)