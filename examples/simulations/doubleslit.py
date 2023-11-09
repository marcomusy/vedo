"""Simulation of the double slit experiment.
(Any number of slits of any geometry can be added)
Slit sources are placed on the plane shown as a thin grid"""
# Can simulate the 'Arago spot', the bright point at the center of
#  a circular object shadow (https://en.wikipedia.org/wiki/Arago_spot).
from vedo import *

#########################################
lambda1 = 680e-9  # red wavelength 680nm
width = 10e-6     # slit width in m
D = 0.1           # screen distance in m
#########################################

# create the slits as a set of individual coherent point-like sources
n = 10  # nr of elementary sources in slit (to control precision).
slit1 = list(zip([0]*n, np.arange(0,n)*width/n, [0]*n))  # source points inside slit1
slit2 = list(slit1 + np.array([1e-5, 0, 0]))             # a shifted copy of slit 1
slits = slit1 + slit2
# slits += list(slit1 + array([-2e-5, 1e-5, 0]))  # add another copy of slit1
# slits = [(cos(x)*4e-5, sin(x)*4e-5, 0) for x in arange(0,2*np.pi, .1)] # Arago spot
# slits = Grid(s=[1e-4,1e-4], res=[9,9]).vertices # a square lattice

screen = Grid(pos=[0, 0, -D], s=[0.1,0.1], lw=0, res=[200,50]).wireframe(False)

# Compute the image on the far screen
k = 0.0 + 1j * 2 * np.pi / lambda1  # complex wave number
norm = len(slits) * 5e5
amplitudes = []
verts = screen.vertices
for i, x in enumerate(verts):
    psi = 0
    for s in slits:
        r = mag(x - s)
        psi += np.exp(k * r) / r
    psi2 = np.real(psi * np.conj(psi))  # psi squared
    amplitudes.append(psi2)
    verts[i] = x + [0, 0, psi2 / norm]
screen.cmap("hot", amplitudes)

plt = Plotter(title="The Double Slit Experiment", axes=9, bg="black")
plt += [screen, __doc__]
plt += Points(np.array(slits) * 200).color("w")  # slits scale magnified by factor 200
plt += Grid(s=[0.1,0.1], res=[6,6],).color("w",0.1)
plt += Line([0, 0, 0], [0, 0, -D],).color("w",0.1)
plt += Text3D("source plane", pos=[-0.04, -0.05, 0], s=0.002).c("gray")
plt += Text3D("detector plane D = "+str(D)+" m", pos=[-.04,-.05,-D+.001], s=.002).c("gray")
plt.show(zoom=1.15).close()
