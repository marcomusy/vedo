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
line = Line(pts).lw(2)
line.cmap("winter", vel).add_scalarbar("speed")
line.add_shadow("x",   3, alpha=0.2)
line.add_shadow("z", -25, alpha=0.2)

pt = Point(pts[0]).color("red4").ps(12)
pt.add_trail(lw=4).add_shadow("x", 3, alpha=0.5)
pt.trail.add_shadow("x", 3, alpha=0.5)

def loop_func(event): # move the point
    if len(pts) > 0:
        pos = pts.pop(0)
        pt.pos(pos).update_trail().update_shadows()
        plt.render()

plt = Plotter(axes=dict(xygrid=False))
plt.add_callback("timer", loop_func)
plt.timer_callback("start")
plt.show(line, pt, __doc__, viewup="z")
plt.close()
