"""
Quantum Tunnelling effect using 4th order Runge-Kutta method
with arbitrary potential shape.
The animation shows the evolution of a particle of well defined momentum
(hence undefined position) in a box hitting a potential barrier.
The wave function is forced to be zero at the box walls (line 23)."""
print(__doc__)
import numpy as np
from vedo import Plotter, Tube, Line, dataurl, settings

dt = 0.004  # time step
x0 = 5      # peak initial position
s0 = 0.75   # uncertainty on particle position
k0 = 10     # initial momentum of the wave packet
Vmax = 0.2  # height of the barrier (try 0 for particle in empty box)

N = 300  # number of points
size = 20.0  # x span [0, size]
x = np.linspace(0, size, N+2)  # we will need 2 extra points for the box wall

V = Vmax * (np.abs(x - 11) < 0.5) - 0.01  # simple square barrier potential

Psi = np.sqrt(1 / s0) * np.exp(-1 / 2 * ((x - x0) / s0) ** 2 + 1j * x * k0)  # wave packet

dx2 = ((x[-1] - x[0]) / (N+2)) ** 2 * 400  # dx**2 step, scaled
nabla2psi = np.zeros(N + 2, dtype=complex)


def f(psi):
    # a smart numpy way to calculate the second derivative in x:
    nabla2psi[1 : N + 1] = (psi[0:N] + psi[2 : N+2] - 2 * psi[1 : N+1]) / dx2
    return 1j * (nabla2psi - V * psi)  # this is the RH of Schroedinger equation!


def d_dt(psi):  # find Psi(t+dt)-Psi(t) /dt with 4th order Runge-Kutta method
    k1 = f(psi)
    k2 = f(psi + dt / 2 * k1)
    k3 = f(psi + dt / 2 * k2)
    k4 = f(psi + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

settings.allowInteraction = True

plt = Plotter(interactive=False, axes=2, size=(1000,500))
plt.xtitle = ""
plt.ytitle = "|\Psi(x,t)|\^2"

bck = plt.load(dataurl+"images/schrod.png").scale(0.015).pos([0, 0, -0.5])
barrier = Line(np.stack((x, V * 15), axis=1), c="dr", lw=3)

lines = []
for j in range(150):
    for i in range(500):
        Psi += d_dt(Psi) * dt  # integrate for a while
    A = np.real(Psi * np.conj(Psi)) * 1.5  # psi squared, probability(x)
    coords = np.stack((x, A, np.zeros_like(x)), axis=1)
    Aline = Tube(coords, c="db", r=0.08)
    plt.show(Aline, barrier, bck, zoom=2)
    lines.append(Aline)
    if plt.escaped: break # if ESC is hit during the loop

plt.interactive().close()
