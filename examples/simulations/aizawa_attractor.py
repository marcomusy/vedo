"""
Interactive particle swarm on the Aizawa strange attractor.
The script integrates many nearby initial conditions in the Aizawa dynamical
system and displays them as a live 3D point cloud. Particle colors
are derived from instantaneous speed, making the slower recirculating regions
and faster streaming regions of the flow easier to read visually.
"""
import numpy as np
from vedo import Axes, Plotter, Points, color_map


def deriv(states, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    """Evaluate the Aizawa vector field for a batch of particle states."""
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    x2 = x * x; y2 = y * y; z2 = z * z; z3 = z2 * z
    vx = (z - b) * x - d * y
    vy = d * x + (z - b) * y
    vz = c + a * z - z3 / 3.0 - (x2 + y2) * (1.0 + e * z) + f * z * x2 * x
    return np.column_stack([vx, vy, vz])


def rk4_step(states, dt):
    """Advance all particles by one fourth-order Runge-Kutta integration step."""
    slope_1 = deriv(states)
    slope_2 = deriv(states + 0.5 * dt * slope_1)
    slope_3 = deriv(states + 0.5 * dt * slope_2)
    slope_4 = deriv(states + dt * slope_3)
    return states + (dt / 6.0) * (slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4)

def cloud_colors(states, cmap_name):
    """Map particle speed to RGB colors using a named Matplotlib colormap."""
    speed = np.linalg.norm(deriv(states), axis=1)
    l, h = np.percentile(speed, [5, 95])
    mapped_colors = color_map(speed, name=cmap_name, vmin=l, vmax=h)
    return np.clip(mapped_colors * 255.0, 0, 255).astype(np.uint8)


# MAIN ################################################################
particle_count = 50_000
time_step = 0.005
render_scale = 3.0
seed_center = np.array([0.1, 0.0, 0.0])

states = seed_center + np.random.normal(0, 0.05, (particle_count, 3))
render_positions = states * render_scale
swarm_points = Points(render_positions, r=3)
swarm_points.pointcolors = cloud_colors(states, "managua")

def loop_func(_event):
    global states
    for _ in range(3):
        states = rk4_step(states, time_step)
    render_positions = states * render_scale
    swarm_points.points = render_positions
    # swarm_points.pointcolors = cloud_colors(states, "managua")
    plotter.render()

plotter = Plotter(bg="#151325", bg2="#252335", size=(1200,1200))
axes = Axes(xrange=[-5,5], yrange=[-5,5]).shift([0,0,-2])
plotter.show(swarm_points, axes, viewup='z', interactive=False)
plotter.add_callback("timer", loop_func, enable_picking=False)
plotter.timer_callback("start", dt=10)
plotter.interactive().close()
