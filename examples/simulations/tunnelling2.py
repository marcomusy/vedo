"""Quantum Tunnelling effect using 4th order Runge-Kutta
method with arbitrary potential shape.
The animation shows the evolution of a particle of relatively well defined
momentum (hence undefined position) in a box hitting a potential barrier."""
print(__doc__)
import numpy as np
from vedo import Plotter, Line, dataurl, settings

Nsteps = 250  # number of steps in time
N = 300       # number of points in space
dt = 0.004    # time step
x0 = 6        # peak initial position
s0 = 0.75     # uncertainty on particle position
k0 = 10       # initial momentum of the wave packet
Vmax = 0.2    # height of the barrier (try 0 for particle in empty box)
size = 20.0   # x axis span [0, size]

# Uncomment below for more examples of the potential V(x).
x = np.linspace(0, size, N+2)
V = 0.15 * np.sin(1.5 * (x - 7))     # particle hitting a sinusoidal barrier
# V = Vmax*(np.abs(x-11) < 0.5)-.01  # simple square barrier potential
# V = -1.2*(np.abs(x-11) < 1.7)-.01  # a wide square well potential
# V = 0.008*(x-10)**2                # elastic potential well
# V = -0.1*(x-10)                    # particle on a slope bouncing back

Psi = np.sqrt(1/s0) * np.exp(-1/2 * ((x-x0)/s0)**2 + 1j*x*k0)  # wave packet

dx2 = ((x[-1] - x[0]) / (N+2))**2 * 400  # dx**2 step, scaled
nabla2psi = np.zeros(N+2, dtype=complex)

def f(psi):
    # a smart numpy way to calculate the second derivative in x:
    nabla2psi[1 : N+1] = (psi[0:N] + psi[2 : N+2] - 2 * psi[1 : N+1]) / dx2
    return 1j * (nabla2psi - V*psi)  # this is the RH of Schroedinger equation!

def d_dt(psi):  # find Psi(t+dt)-Psi(t) /dt with 4th order Runge-Kutta method
    k1 = f(psi)
    k2 = f(psi + dt / 2 * k1)
    k3 = f(psi + dt / 2 * k2)
    k4 = f(psi + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

settings.allowInteraction = True

plt = Plotter(interactive=False)
bck = plt.load(dataurl+"images/schrod.png").alpha(.3).scale(.0256).pos([0,-5,-.1])
barrier = Line(np.stack((x, V*15, np.zeros_like(x)), axis=1), c="black", lw=2)
box = bck.box().c('black')

lines = []
for i in range(0, Nsteps):
    for j in range(500):
        Psi += d_dt(Psi) * dt  # integrate for a while before showing things
    A = np.real(Psi * np.conj(Psi)) * 1.5  # psi squared, probability(x)
    coords = np.stack((x, A), axis=1)
    Aline = Line(coords, c="db", lw=3)
    plt.show(barrier, bck, Aline, box).remove(Aline)
    lines.append([Aline, A])   # store objects

# now show the same lines along z representing time
plt.actors= [] # clean up internal list of objects to show
plt.camera.Elevation(20)
plt.camera.Azimuth(20)
bck.alpha(1)
for i in range(Nsteps):
    p = [0, 0, i*size/Nsteps]  # shift along z
    l, a = lines[i]
    l.cmap("gist_earth_r", a)
    plt += [box, bck, l.pos(p), barrier.clone().alpha(0.3).pos(p)]
    plt.show()

plt.interactive().close()
