"""Simulation of a block connected to a spring in a viscous medium"""
from vedo import *
from vedo.player_animation import PlayerAnimation

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


simulated_step = -1
def update_plot(i: int):
    global simulated_step
    diff = i - simulated_step
    if diff > 5:
        pb = ProgressBar(0, diff, c="blue")
    else:
        pb = None
    while i > simulated_step:
        if pb:
            new_diff = i - simulated_step
            pb.print(str(diff - new_diff))
        simulated_step = simulated_step + 1
        update_simulation(simulated_step)
    show_history(i)

animation = PlayerAnimation(
    func=update_plot,
    axes=0,
)

animation.plotter += Box(pos=(0, -0.1, 0), size=(2.0, 0.02, 0.5))  # floor
animation.plotter += Box(pos=(-0.82, 0.15, 0), size=(0.04,0.50,0.3))  # wall

block = Cube(pos=x, side=0.2, c="tomato")
spring = Spring(sx0, x, r1=0.06, thickness=0.01)
animation.plotter += [block, spring, __doc__]

history_x = []
history_sx0 = []

def update_simulation(i: int):
    global v, x, history_x, history_sx0
    F = -k * (x - xr) - b*v  # Force and friction
    a = F / m  # acceleration
    v = v + a*dt  # velocity
    x = x + v*dt + 1/2 * a * dt**2  # position
    
    history_x.append(x.copy())
    history_sx0.append(sx0.copy())
    
def show_history(i: int):
    global history_x, history_sx0
    # update block position and trail
    block.pos(history_x[i])  
    # stretch helix accordingly
    spring.stretch(history_sx0[i], history_x[i])

animation.plotter.show(
    interactive=False,
)
animation.set_val(0)
animation.plotter.interactive().close()