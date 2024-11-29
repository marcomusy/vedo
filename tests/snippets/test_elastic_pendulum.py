"""Simulate an elastic pendulum.
The trail is colored according to the velocity."""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from vedo import Plotter, Axes, Sphere, Spring, Image, mag, sin, cos
from vedo.addons import ProgressBarWidget


a = 2.0   # length of the pendulum
m = 0.5   # mass
k = 10.0  # constant of the spring
g = 9.81  # gravitational constant

# Define the system of ODEs
def system(t, z):
    x, dx_dt, y, dy_dt = z  # z = [x, x', y, y']
    dxdot_dt = (a+x) * dy_dt**2 - k/m * x + g * cos(y)
    dydot_dt = -2/(a+x) * dx_dt * dy_dt - g/(a+x) * sin(y)
    return [dx_dt, dxdot_dt, dy_dt, dydot_dt]


# Initial conditions: x(0), x'(0), y(0), y'(0)
initial_conditions = [0.0,   0.0,  0.4,   0.0]

# Time span for the solution
t_span = (0, 12)
t_eval = np.linspace(t_span[0], t_span[1], 500) # range to evaluate solution

# Solve the system numerically
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval)
t_values = solution.t
elong_values = solution.y[0]
theta_values = solution.y[2]

# Plot the results using matplotlib as a graph
fig = plt.figure()
plt.plot(t_values, elong_values, label="elongation(t)")
plt.plot(t_values, theta_values, label="angle(t)")
plt.xlabel("Time")
plt.legend()

# Animate the system using the solution of the ODE
plotter = Plotter(bg="blackboard", bg2="blue1", interactive=False)
pbw  = ProgressBarWidget(len(t_values))
axes = Axes(xrange=[-2, 2], yrange=[-a*2, 1], xygrid=0, xyframe_line=0, c="w")
img  = Image(fig).clone2d("top-right", size=0.5)
sphere = Sphere(r=0.3, c="red5").add_trail(c="k5", lw=4)
plotter.show(axes, sphere, img, pbw, __doc__)

for elong, theta in zip(elong_values, theta_values):
    x =  (a + elong) * sin(theta)
    y = -(a + elong) * cos(theta)
    spring = Spring([0, 0], [x, y])
    sphere.pos([x, y]).update_trail()

    # color the trail according to the lenght of each segment
    v = sphere.trail.vertices
    lenghts1 = np.array(v[1:])
    lenghts2 = np.array(v[:-1])
    lenghts = mag(lenghts1 - lenghts2) # lenght of each segment
    lenghts = np.append(lenghts, lenghts[-1])
    sphere.trail.cmap("Blues_r", lenghts, vmin=0, vmax=0.1)

    plotter.remove("Spring").add(spring).render()
    pbw.update()  # update progress bar

plotter.interactive()
