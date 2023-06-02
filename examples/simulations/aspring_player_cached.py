"""Simulation of a block connected to a spring in a viscous medium"""
from vedo import *
from vedo.applications import PlayerAnimationCached

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
offx = vector(0, 0.3, 0)

actors = {
    "floor" : Box(pos=(0, -0.1, 0), size=(2.0, 0.02, 0.5)),
    "wall" : Box(pos=(-0.82, 0.15, 0), size=(0.04,0.50,0.3)),
    "block": Cube(pos=x, side=0.2, c="tomato"),
    "spring": Spring(sx0, x, r1=0.06, thickness=0.01),
    "doc":__doc__
}

def update_simulation(i: int) -> tuple:
    global v, x
    F = -k * (x - xr) - b*v  # Force and friction
    a = F / m  # acceleration
    v = v + a*dt  # velocity
    x = x + v*dt + 1/2 * a * dt**2  # position
    return x, sx0
    
def show_history(state: tuple) -> None:
    x, sx0 = state
    # update block position and trail
    actors["block"].pos(x)  
    # stretch helix accordingly
    actors["spring"].stretch(sx0, x)

animation = PlayerAnimationCached(
    simulation_func=update_simulation,
    show_func=show_history,
    actors=actors,
    size=(1050, 600),
)