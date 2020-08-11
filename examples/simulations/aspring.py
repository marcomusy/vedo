"""Simulation of a block connected 
to a spring in a viscous medium.
"""
from vedo import *

vp = Plotter(interactive=0, axes=0)

L = 0.1  # spring x position at rest
x0 = 0.85  # initial x-coordinate of the block
k = 25  # spring constant
m = 20  # block mass
b = 0.5  # viscosity friction (proportional to velocity)
dt = 0.15  # time step

# initial conditions
v = vector(0, 0, 0.2)
x = vector(x0, 0, 0)
xr = vector(L, 0, 0)
sx0 = vector(-0.8, 0, 0)
offx = vector(0, 0.3, 0)

vp += Box(pos=(0, -0.1, 0), length=2.0, width=0.02, height=0.5)  # surface
vp += Box(pos=(-0.82, 0.15, 0), length=0.04, width=0.50, height=0.3)  # wall

block = Cube(pos=x, side=0.2, c="tomato")
block.addTrail(offset=[0, 0.2, 0], alpha=0.6, lw=2, n=500)
spring = Spring(sx0, x, r=0.06, thickness=0.01)
vp += [block, spring, Text2D(__doc__)]

pb = ProgressBar(0, 300, c="r")
for i in pb.range():
    F = -k * (x - xr) - b * v  # Force and friction
    a = F / m  # acceleration
    v = v + a * dt  # velocity
    x = x + v * dt + 1 / 2 * a * dt ** 2  # position

    block.pos(x)  # update block position and trail
    spring.stretch(sx0, x)  # stretch helix accordingly

    vp.show(elevation=0.1, azimuth=0.1)
    pb.print()

vp.show(interactive=1)
