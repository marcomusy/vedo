"""The Lorenz attractor is a set of chaotic solutions of
a particular system of ordinary differential equations"""
from vedo import *

p = (25, -10, -7)  # starting point (initial condition)
dt = 0.01          # time step

# Define the ODE system to integrate (Lorenz equations)
pts, vel = [], []
for t in np.arange(0, 20, dt):
    x, y, z = p
    v = np.array([-8/3*x+y*z, -10*(y-z), -y*x+28*y-z])
    p = p + v * dt
    pts.append(p)
    vel.append(mag(v))

# Plot the trajectory in 3D space
line = Line(pts).lw(2).cmap("winter", vel).add_scalarbar("speed")
line.add_shadow("x",   3, alpha=0.2)
line.add_shadow("z", -25, alpha=0.2)
pt = Point(pts[0], c="red4", r=12).add_trail(lw=4)

def loop_func(event):
    global i
    if i < len(pts):
        pt.pos(pts[i]).update_trail() # move the point
        plt.render()
        i += 1
    else:
        plt.timer_callback("stop", tid) # stop the timer

i = 0
plt = Plotter(axes=dict(xygrid=False))
plt.show(line, pt, __doc__, viewup="z", interactive=False)
plt.add_callback("timer", loop_func)
tid = plt.timer_callback("start")
plt.interactive().close()

