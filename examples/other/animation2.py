"""
This example shows how to animate simultaneously various objects
by specifying event times and durations of the visual effects.
"""
from vedo import *
from vedo.applications import Animation

s = load(dataurl+"bunny.obj").subdivide().normalize()

vp = Animation()
vp.timeResolution = 0.02  # secs

vp.switchOn(s)

# no need to repeat t=1, duration=3 in changeLighting and changeColor
vp.meshErode(corner=0, t=1, duration=3).changeLighting("glossy").changeColor("v")

cam1 = orientedCamera(backoffVector=(0, 0, -1), backoff=8)
cam2 = orientedCamera(backoffVector=(1, 1,  1), backoff=8)

vp.moveCamera(cam1, cam2, t=0, duration=4)

vp.play()
