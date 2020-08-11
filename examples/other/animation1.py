"""
This example shows how to animate simultaneously various objects
by specifying event times and durations of the effects
"""
from vedo import *
from vedo.applications import Animation

sp = Sphere(r=0.5).cutWithPlane(origin=(0.15,0,0)).lw(0.1)
cu = Cube().pos(-2,0,0)
tr = Torus().pos(1,0,0).rotateY(80)

vp = Animation()
vp.showProgressBar = True
vp.timeResolution = 0.025  # secs

vp.fadeIn([cu, tr], t=0, duration=0.2)
vp.fadeIn(sp, t=1, duration=2)

vp.move(sp, (2,0,0), style="linear")
vp.rotate(sp, axis="y", angle=180)

vp.fadeOut(sp, t=3, duration=2)
vp.fadeOut(tr, t=4, duration=1)

vp.scale(cu, 0.1, t=5, duration=1)

vp.totalDuration = 4 # can shrink/expand total duration

vp.play()
