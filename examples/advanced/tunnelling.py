# Quantum Tunnelling effect using 4th order Runge-Kutta method
# with arbitrary potential shape.
# The animation shows the evolution of a particle of well defined momentum
# (hence undefined position) in a box hitting a potential barrier.
#
from __future__ import division, print_function
from vtkplotter import Plotter, ProgressBar
import numpy as np

N = 400      # number of points
dt = 0.005   # time step
size = 20    # x span [0, size]
x0 = 5       # peak initial position
Vmax = 1.2   # height of the barrier (try 0 for particle in empty box)

x = np.linspace(0, size, N+2) 
V = Vmax*( np.abs(x - 12) < 0.5 )-0.1  # barrier potential

Psi = np.exp(-4*(x-x0) *((x-x0)+10j))  # initial wave function with velocity 10j

def f(psi):
    nabla2psi = np.zeros(N+2, dtype=np.complex) 
    # a smart way to calculate the second derivative in x:
    nabla2psi[1:N+1] = (psi[0:N]+psi[2:N+2] -2*psi[1:N+1])/2 
    return (nabla2psi - V*psi) * 1j    # this is the RH of Schroedinger equation!

def dPsi(psi): # find Psi(t+dt)-Psi(t) with 4th order Runge-Kutta method
    k1 = f(psi)
    k2 = f(psi +dt/2*k1)    
    k3 = f(psi +dt/2*k2)    
    k4 = f(psi +dt  *k3)    
    dpsi = (k1 + 2 * k2 + 2 * k3 + k4) *dt / 6
    return dpsi

vp = Plotter(interactive=0, axes=2, verbose=0)
vp.xtitle = ''
vp.ytitle = 'Psi^2(x,t)'
vp.ztitle = ''
nx = len(x)

bck = vp.load('data/images/schrod.png').scale(0.012).pos([0,0,-.5])
barrier = vp.line(list(zip(x, V*2,  [0]*nx)), c='dr', lw=3)

lines = []
pb = ProgressBar(0, 200, c='blue', ETA=0)
for j in pb.range():	

    for i in range(500): Psi += dPsi(Psi) # integrate for a while
    
    A = np.real( Psi*np.conj(Psi) )*2     # psi squared, probability(x)
    coords = list(zip(x, A,  [0]*nx))
    Aline = vp.line(coords, c='db', tube=True, lw=.08)
    vp.show([Aline, barrier, bck])
    lines.append(Aline)
    pb.print()

# now show the same lines along z representing time
vp.clear()
vp.camera.Elevation(20)
vp.camera.Azimuth(20)

for i,l in enumerate(lines):
    p = [0, 0, 20*i/len(lines)] # shift along z
    vp.render([l.pos(p), barrier.clone().pos(p)], resetcam=True)

vp.show(interactive=1)










