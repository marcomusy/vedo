"""
This example shows how to animate simultaneously various objects
by specifying event times and durations of the effects
"""
from vedo import *
from vedo.applications import Animation

sp = Sphere(r=0.5).cutWithPlane(origin=(0.15,0,0)).lw(0.1)
cu = Cube().pos(-2,0,0)
tr = Torus().pos(1,0,0).rotateY(80)

plt = Animation()
plt.showProgressBar = True
plt.timeResolution = 0.025  # secs
plt.totalDuration = 4 # can shrink/expand total duration

plt.fadeIn([cu, tr], t=0, duration=0.2)
plt.fadeIn(sp, t=1, duration=2)

plt.move(sp, (2,0,0), style="linear")
plt.rotate(sp, axis="y", angle=180)

plt.fadeOut(sp, t=3, duration=2)
plt.fadeOut(tr, t=4, duration=1)

plt.scale(cu, 0.1, t=5, duration=1)
plt.play()
