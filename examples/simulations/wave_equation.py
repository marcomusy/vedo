"""Simulate a discrete collection of oscillators
We will use this as a model of a vibrating string and
compare two methods of integration: Euler and Runge-Kutta4.
For too large values of dt the simple Euler can diverge."""
# To model 'N' oscillators, we will use N+2 Points, numbered
# 0, 1, 2, 3, ... N+1.  Points 0 and N+1 are actually the boundaries.
# We will keep them fixed, but adding them in as if they were
# masses makes the programming easier.
# Adapted from B.Martin (2009) http://www.kcvs.ca/martin by M.Musy
from vedo import Plotter, ProgressBar, Point, dataurl, settings
import numpy as np

####################################################
N = 400  # Number of coupled oscillators
dt = 0.5  # Time step
Nsteps = 1200  # Number of steps in the simulation


####################################################
# Initial positions
####################################################
x = np.array(list(range(N + 2)))
z = np.zeros(N + 2, float)
y = np.zeros(N + 2, float)  # y[p] is the position of particle p

for p in x:  # p is particle number along x axis
    y[p] = 100 * np.sin(p/15) * np.exp(-p/50)


####################################################
# Initial velocities
####################################################
v = np.zeros(N + 2, float)
# or you can give one specific particle a kick:
# v[40] = 50


####################################################
# Integrate forward
####################################################
# Acceleration function for the simple harmonic oscillator
def accel(y, v, t):
    a = np.zeros(N + 2, float)  # acceleration of particles
    # for p in range(1,N+1): a[p] = -(y[p]-y[p-1]) -(y[p]-y[p+1]) #slower
    a[1 : N + 1] = -(y[1 : N + 1] - y[0:N]) - (y[1 : N + 1] - y[2 : N + 2])  # faster
    return a


def rk4(y, v, t, dt):  # 4th Order Runge-Kutta
    yk1 = dt * v
    vk1 = dt * accel(y, v, t)

    yk2 = dt * (v + vk1 / 2)
    vk2 = dt * accel(y + yk1 / 2, v + vk1 / 2, t + dt / 2)

    yk3 = dt * (v + vk2 / 2)
    vk3 = dt * accel(y + yk2 / 2, v + vk2 / 2, t + dt / 2)

    yk4 = dt * (v + vk3)
    vk4 = dt * accel(y + yk3, v + vk3, t + dt)

    ynew = y + (yk1 + 2 * yk2 + 2 * yk3 + yk4) / 6
    vnew = v + (vk1 + 2 * vk2 + 2 * vk3 + vk4) / 6
    return ynew, vnew


def euler(y, v, t, dt):  # simple euler integrator
    vnew = v + accel(y, v, t) * dt
    ynew = y + vnew * dt + 1 / 2 * accel(y, vnew, t) * dt ** 2
    return ynew, vnew


positions_eu, positions_rk = [], []
y_eu, y_rk = np.array(y), np.array(y)
v_eu, v_rk = np.array(v), np.array(v)
t = 0
pb = ProgressBar(0, Nsteps, c="blue", ETA=0)
for i in pb.range():
    y_eu, v_eu = euler(y_eu, v_eu, t, dt)
    y_rk, v_rk = rk4(y_rk, v_rk, t, dt)
    t += dt
    positions_eu.append(y_eu)  # store result of integration
    positions_rk.append(y_rk)
    pb.print("Integrate: RK-4 and Euler")


####################################################
# Visualize the result
####################################################
settings.allowInteraction = True

plt = Plotter(interactive=False, axes=2)  # choose axes type nr.2

for i in x:
    plt += Point([i, 0, 0], c="green", r=6)
pts_actors_eu = plt.actors  # save a copy of the actors list
pts_actors_eu[0].legend = "Euler method"

plt.actors = []  # clean up the list

for i in x:
    plt += Point([i, 0, 0], c="red", r=6)
pts_actors_rk = plt.actors  # save a copy of the actors list
pts_actors_rk[0].legend = "Runge-Kutta4"

# merge the two lists and set it as the current actors
plt.actors = pts_actors_eu + pts_actors_rk

# let's also add a fancy background image from wikipedia
plt.load(dataurl+"images/wave_wiki.png").alpha(0.8).scale(0.4).pos(0,-100,-20)
plt += __doc__

pb = ProgressBar(0, Nsteps, c="red", ETA=1)
for i in pb.range():
    y_eu = positions_eu[i]  # retrieve the list of y positions at step i
    y_rk = positions_rk[i]
    for j, act in enumerate(pts_actors_eu):
        act.pos(j, y_eu[j], 0)
    for j, act in enumerate(pts_actors_rk):
        act.pos(j, y_rk[j], 0)
    if i%10 ==0:
        plt.show()
    if plt.escaped: break  # if ESC is hit during the loop
    pb.print("Moving actors loop")

plt.interactive().close()
