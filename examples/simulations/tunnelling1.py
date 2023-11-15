"""Quantum tunneling using 4th order Runge-Kutta method"""
import numpy as np
from vedo import Plotter, Line

N = 300      # number of points
dt = 0.004   # time step
x0 = 5       # peak initial position
s0 = 0.75    # uncertainty on particle position
k0 = 10      # initial momentum of the wave packet
Vmax = 0.2   # height of the barrier (try 0 for particle in empty box)
size = 20.0  # x span [0, size]


def f(psi):
    nabla2psi = np.zeros(N+2, dtype=complex)
    dx2 = ((x[-1] - x[0]) / (N+2))**2 * 400  # dx**2 step, scaled
    nabla2psi[1 : N+1] = (psi[0:N] + psi[2 : N+2] - 2 * psi[1 : N+1]) / dx2
    return 1j * (nabla2psi - V * psi)  # this is the RHS of Schroedinger equation


def d_dt(psi):  # find Psi(t+dt)-Psi(t) /dt with 4th order Runge-Kutta method
    k1 = f(psi)
    k2 = f(psi + dt / 2 * k1)
    k3 = f(psi + dt / 2 * k2)
    k4 = f(psi + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


x = np.linspace(0, size, N+2)  # we will need 2 extra points for the box wall
V = Vmax * (np.abs(x-11) < 0.5) - 0.01  # simple square barrier potential

Psi = np.sqrt(1/s0) * np.exp(-1/2 * ((x-x0)/s0)**2 + 1j * x * k0)  # wave packet
zeros = np.zeros_like(x)

plt = Plotter(interactive=False, size=(1000,500))

barrier = Line(np.c_[x, V * 15]).c("red3").lw(3)
wpacket = Line(np.c_[x,  zeros]).c('blue4').lw(2)
plt.show(barrier, wpacket, __doc__, zoom=2)

for j in range(150):
    for i in range(500):
        Psi += d_dt(Psi) * dt                # integrate for a while
    amp = np.real(Psi * np.conj(Psi)) * 1.5  # psi squared, probability(x)
    wpacket.vertices = np.c_[x, amp, zeros]  # update vertices
    plt.render()

plt.interactive().close()
