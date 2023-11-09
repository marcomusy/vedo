"""Quantum Tunnelling effect using 4th order Runge-Kutta
method with arbitrary potential shape."""
from vedo import *

nsteps = 150  # number of steps in time
n = 300       # number of points in 1d space
dt = 0.004    # time step
x0 = 6        # peak initial position
s0 = 0.75     # uncertainty on particle position
k0 = 10       # initial momentum of the wave packet
Vmax = 0.2    # height of the barrier (try 0 for particle in empty box)
size = 20.0   # x axis span [0, size]

# Uncomment below for more examples of the potential V(x).
x = np.linspace(0, size, n+2)
V = 0.15 * np.sin(1.5 * (x - 7))       # particle hitting a sinusoidal barrier
# V = Vmax*(np.abs(x-11) < 0.5)-0.01   # simple square barrier potential
# V = -0.5*(np.abs(x-11) < 1.7)-0.01   # a wide square well potential
# V = 0.008*(x-10)**2                  # elastic potential well
# V =  0.05*(x-10)                     # particle on a slope bouncing back
# V =  0.0 * x                         # free particle

Psi = np.sqrt(1/s0) * np.exp(-1/2 * ((x-x0)/s0)**2 + 1j*x*k0)  # wave packet

dx2 = ((x[-1] - x[0]) / (n+2))**2 * 400  # dx**2 step, scaled
nabla2psi = np.zeros(n+2, dtype=complex)

def f(psi):
    # a smart numpy way to calculate the second derivative in x:
    nabla2psi[1 : n+1] = (psi[0:n] + psi[2 : n+2] - 2 * psi[1 : n+1]) / dx2
    return 1j * (nabla2psi - V*psi)  # this is the RH of Schroedinger equation!

def d_dt(psi):  # find Psi(t+dt)-Psi(t) /dt with 4th order Runge-Kutta method
    k1 = f(psi)
    k2 = f(psi + dt / 2 * k1)
    k3 = f(psi + dt / 2 * k2)
    k4 = f(psi + dt * k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


plt = Plotter(interactive=False)
pic = Image(dataurl+"images/schrod.png").pos(0, -5, -0.1).scale(0.0255)
barrier = Line(np.stack((x, V*15, np.zeros_like(x)), axis=1), c="black", lw=2)
barrier.name = "barrier"
plt.show(pic, barrier, __doc__)

lines = []
for i in range(nsteps):
    for j in range(500):
        Psi += d_dt(Psi) * dt  # integrate for a while before showing things
    A = np.real(Psi * np.conj(Psi)) * 1.5  # psi squared, probability(x)
    coords = np.stack((x, A), axis=1)
    Aline = Line(coords).color("db").linewidth(3)
    lines.append([Aline, A])   # store objects
    plt.remove("Line").add(Aline).render()

# now show the same lines along z representing time
plt.objects= [] # clean up internal list of objects to show
plt.elevation(20).azimuth(20)

barrier.color('black', 0.3)
barrier_end = barrier.clone().pos([0,0,20])
rib = Ribbon(barrier, barrier_end).c("black",0.1)
plt.add(rib)
plt.reset_camera()

for i in range(nsteps):
    p = [0, 0, i*size/nsteps]  # shift along z
    line, A = lines[i]
    line.cmap("gist_earth_r", A).pos(p)
    plt.add(pic, line).render()

plt.interactive().close()
