"""Simulate a discrete collection of oscillators
We will use this as a model of a vibrating string and
compare two methods of integration: Euler (red) and Runge-Kutta4 (green).
For too large values of dt the simple Euler will diverge."""
# To model 'N' oscillators, we will use N+2 Points, numbered
# 0, 1, 2, 3, ... N+1.  Points 0 and N+1 are actually the boundaries.
# We will keep them fixed, but adding them in as if they were
# masses makes the programming easier.
# Adapted from B.Martin (2009) http://www.kcvs.ca/martin by M.Musy
from vedo import *

####################################################
N = 400        # Number of coupled oscillators
dt = 0.5       # Time step
nsteps = 2000  # Number of steps in the simulation


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
    a[1 : N+1] = -(y[1 : N+1] - y[0:N]) - (y[1 : N+1] - y[2 : N+2])
    return a


def runge_kutta4(y, v, t, dt):  # 4th Order Runge-Kutta
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
for i in progressbar(nsteps, c="b", title="integrating RK4 and Euler"):
    y_eu, v_eu = euler(y_eu, v_eu, t, dt)
    y_rk, v_rk = runge_kutta4(y_rk, v_rk, t, dt)
    t += dt
    positions_eu.append(y_eu)  # store result of integration
    positions_rk.append(y_rk)

####################################################
# Visualize the result
####################################################
plt = Plotter(interactive=False, axes=2, size=(1400,1000))

line_eu = Line([0,0,0], [len(x)-1,0,0], res=len(x)).c("red5").lw(5)
plt += line_eu

line_rk = Line([0,0,0], [len(x)-1,0,0], res=len(x)).c("green5").lw(5)
plt += line_rk

# let's also add a fancy background image from wikipedia
img = dataurl + "images/wave_wiki.png"
plt += Image(img).alpha(0.8).scale(0.4).pos(0,-100,-1)
plt += __doc__
plt.show(zoom=1.5)

for i in progressbar(nsteps, title="visualize the result", c='y'):
    if i%10 != 0:
        continue
    y_eu = positions_eu[i]  # retrieve the list of y positions at step i
    y_rk = positions_rk[i]

    pts = line_eu.vertices
    pts[:,1] = y_eu
    line_eu.vertices = pts

    pts = line_rk.vertices
    pts[:,1] = y_rk
    line_rk.vertices = pts
    plt.render()

plt.interactive().close()
