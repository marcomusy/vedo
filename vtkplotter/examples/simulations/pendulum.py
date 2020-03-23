'''
Visualize the phase space of a simple pendulum.
x = starting angle theta
y = starting angular velocity
'''
# From https://www.youtube.com/watch?v=p_di4Zn4wz4
# Overview of differential equations | Chapter 1
# (by 3Blue1Brown)

#Install with:
# pip install vtkplotter

from vtkplotter import *

# Physical constants
g = 9.81 # m/s^2
L = 2    # m
mu = 0.1 # friction 1/s
delta_t = 0.01 # Some time step
t_tot = 50 # seconds total time

# Definition of ODE
def get_theta_dot_dot(theta, theta_dot):
    return -mu * theta_dot - (g/L) * sin(theta)

lines = []
for THETA_0 in arange(0, 3.1415, 0.2):
    for THETA_DOT_0 in arange(4, 9, 1):

        # Solution to the differential equation
        theta = THETA_0
        theta_dot = THETA_DOT_0
        pts = []
        for time in arange(0, t_tot, delta_t):
            theta_dot_dot = get_theta_dot_dot(theta, theta_dot)
            theta     += theta_dot * delta_t
            theta_dot += theta_dot_dot * delta_t
            pts.append([theta, theta_dot])

        l = Line(pts).color(int(THETA_DOT_0))
        lines.append(l)

show(lines, Text2D(__doc__), axes=2)
