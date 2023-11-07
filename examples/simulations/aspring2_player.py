"""Animation of a block attached to a spring"""
from vedo import *
from vedo.applications import AnimationPlayer

L = 0.1    # spring x position at rest
x0 = 0.85  # initial x-coordinate of the block
k = 20     # spring constant
m = 20     # block mass
b = 5      # viscosity friction (proportional to velocity)
dt = 0.15  # time step

v  = vector(0, 0, 0.3)  # initial conditions
x  = vector(x0, 0, 0)
xr = vector(L, 0, 0)
x0 = vector(-0.8, 0, 0)

# Pre-compute the trajectory of the block and store it in a list.
history_x = []
for i in range(200):
    F = -k * (x - xr) - b * v         # force and friction
    a = F / m                         # acceleration
    v = v + a * dt                    # velocity
    x = x + v * dt + 1/2 * a * dt**2  # position
    history_x.append(x)

# Create the objects to be shown in the animation
floor = Box(pos=(0, -0.1, 0), size=(2.0, 0.02, 0.5)).c('yellow2')
wall  = Box(pos=(-0.82, 0.15, 0), size=(0.04, 0.50, 0.3)).c('yellow2')
block = Cube(pos=x, side=0.2).c("tomato")
spring= Spring(x0, x, r1=0.05, thickness=0.005)
text  = Text2D(font="Calco", c='white', bg='k', alpha=1, pos='top-right')

# Create the animation player and it's callback function
def update_scene(i: int):
    # update block and spring position at frame i
    block.pos(history_x[i])
    spring = Spring(x0, history_x[i], r1=0.05, thickness=0.005)
    text.text(f"Frame number {i}\nx = {history_x[i][0]:.4f}")
    plt.remove("Spring").add(spring).render()

plt = AnimationPlayer(update_scene, irange=[0,200], loop=True)
plt += [floor, wall, block, spring, text, __doc__]
plt.set_frame(0)
plt.show()
plt.close()
