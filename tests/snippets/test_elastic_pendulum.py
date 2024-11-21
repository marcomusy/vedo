import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from vedo import Plotter, Axes, Sphere, Spring, Image
from vedo.addons import ProgressBarWidget


a = 2.0   # length of the pendulum
m = 0.5   # mass
k = 10.0  # constant of the spring
g = 9.81  # gravitational constant

# Define the system of ODEs
def system(t, z):
    x, x_dot, y, y_dot = z  # z = [x, x', y, y']
    dx_dt = x_dot
    dy_dt = y_dot
    dx_dot_dt = (a + x) * y_dot ** 2 - k / m * x + g * np.cos(y)
    dy_dot_dt = -2 / (a + x) * x_dot * y_dot - g / (a + x) * np.sin(y)
    return [dx_dt, dx_dot_dt, dy_dt, dy_dot_dt]


# Initial conditions: x(0), x'(0), y(0), y'(0)
initial_conditions = [0.0, 0.0, 0.4, 0.0]

# Time span for the solution
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500) # range to evaluate solution

# Solve the system numerically
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval)
t_values = solution.t
elong_values = solution.y[0]
theta_values = solution.y[2]

# Plot the results
fig = plt.figure()
plt.plot(t_values, elong_values, label="elongation(t)")
plt.plot(t_values, theta_values, label="angle(t)")
plt.xlabel("Time")
plt.legend()

# Use vedo to animate the system
plotter = Plotter(bg="blackboard", bg2="blue1", interactive=False)
pbw = ProgressBarWidget(len(t_values))
axes = Axes(xrange=[-2, 2], yrange=[-a*2, 1], xygrid=0, xyframe_line=0, c="w")
img = Image(fig).clone2d("top-right", size=0.5)
sphere = Sphere(r=0.3, c="red5").add_trail(c="k5", lw=4)
plotter.show(axes, sphere, img, pbw)

for el, theta in zip(elong_values, theta_values):
    x =  (a + el) * np.sin(theta)
    y = -(a + el) * np.cos(theta)
    spring = Spring([0, 0], [x, y])
    sphere.pos([x, y]).update_trail()
    plotter.remove("Spring").add(spring)
    pbw.update()  # update progress bar
    plotter.render()

plotter.interactive()
