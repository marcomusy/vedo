"""Simulation of a block connected to a spring in a viscous medium"""
from vedo import *

L = 0.1    # spring x position at rest
x0 = 0.85  # initial x-coordinate of the block
k = 25     # spring constant
m = 20     # block mass
b = 0.5    # viscosity friction (proportional to velocity)
dt = 0.15  # time step

# initial conditions
v = vector(0, 0, 0.2)
x = vector(x0, 0, 0)
xr = vector(L, 0, 0)
sx0 = vector(-0.8, 0, 0)

plt = Plotter(size=(1050, 600))
plt += Box(pos=(0, -0.1, 0), size=(2.0, 0.02, 0.5))  # floor
plt += Box(pos=(-0.82, 0.15, 0), size=(0.04,0.50,0.3))  # wall

block = Cube(pos=x, side=0.2, c="tomato")
spring = Spring(sx0, x, r1=0.06, thickness=0.01)
plt += [block, spring, __doc__]

def loop_func(event):
    global v, x
    F = -k * (x - xr) - b*v  # Force and friction
    a = F / m  # acceleration
    v = v + a*dt  # velocity
    x = x + v*dt + 1/2 * a * dt**2  # position

    block.pos(x)  # update block position and trail
    spring.stretch(sx0, x)  # stretch helix accordingly
    plt.render()

plt.add_callback("timer", loop_func)
plt.timer_callback("start")
plt.show().close()
