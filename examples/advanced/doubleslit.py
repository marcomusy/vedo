'''
Simulation of the double slit experiment.
Units are meters. Any number of slits of any geometry can be added.
Slit sources are placed on the plane shown as a thin grid
 (as source are in scale, too small to be seen, they are magnified x200).
Can simulate the 'Arago spot', the bright point at the center of 
 a circular object shadow (https://en.wikipedia.org/wiki/Arago_spot).
'''
from numpy import conj, real, pi, array, sin, cos, exp
from vtkplotter import Plotter, arange, mag, grid, line, text

#########################################
lambda1 = 680e-9   # red wavelength 680nm
width   = 10e-6    # slit width in m
D       = 0.1      # screen distance in m
#########################################

# create the slits as a set of individual coherent point-like sources
n = 10 # nr of elementary sources in slit (to control precision).
slit1 = list(zip([0]*n, arange(0,n)*width/n, [0]*n)) # source points inside slit 1
slit2 = list(slit1 + array([1e-5, 0,0]))             # a shifted copy of slit 1
slits = slit1 + slit2  
#slits += list(slit1 + array([-2e-5, 1e-5, 0]))      # add an other copy of slit 1
#slits = [(cos(x)*4e-5, sin(x)*4e-5, 0) for x in arange(0,2*pi, .1)] # Arago spot
#slits = grid(sx=1e-4, sy=1e-4, resx=9, resy=9).coordinates() # a square lattice

vp = Plotter(title='The Double Slit Experiment', axes=0, verbose=0, bg='black')

screen = vp.add(grid(pos=[0,0,-D], sx=0.1, sy=0.1, resx=200, resy=50))
screen.wire(False) # show it as a solid plane (not as wireframe)

k  = 0.0 + 1j * 2*pi/lambda1 # complex wave number
norm = len(slits)*5e+5
amplitudes = []

for i, x in enumerate(screen.coordinates()):
    psi = 0
    for s in slits:
        r = mag(x-s)
        psi += exp( k * r )/r
    psi2 = real( psi * conj(psi) ) # psi squared
    amplitudes.append(psi2)
    screen.point(i, x+[0,0, psi2/norm]) # elevate grid in z

screen.pointColors(amplitudes, cmap='hot')

vp.points(array(slits)*200, c='w')    # slits scale magnified by factor 200

vp.add(grid(sx=0.1, sy=0.1, resx=6, resy=6, c='white/.1')) # add some annotation
vp.add(line([0,0,0], [0,0,-D], c='white/.1'))
vp.add(text("source plane", pos=[-.05,-.053,0], s=.002, c='gray'))
vp.add(text('detector plane D = '+str(D)+' m', pos=[-.05,-.053,-D], s=.002, c='gray'))
vp.add(text(__doc__, c='gray'))

vp.show(zoom=1.1)


