# Black body intensity radiation for different temperatures in [3000K, 9000K] 
# for the visible range of wavelenghts [400nm, 700nm]. Colors are fairly
# well matched to the "jet" and "rainbow" maps in pointColors() method.
#
from vtkplotter import Plotter, arange, exp
from vtkplotter.utils import pointColors

c = 2.9979246e+8
k = 1.3806485e-23 # boltzmann constant
h = 6.6260700e-34 # planck constant

def planck(l, T):
    a = 2 * h * c**2
    b = h*c/(l*k*T)
    return a / ( l**5 * (exp(b)-1) ) * 1e-13 # Planck formula

vp = Plotter(interactive=0, verbose=0, bg='k')
vp.infinity = True # view from infinity (axes are kept orthogonal)
vp.xtitle = ''
vp.ytitle = 'Intensity'
vp.ztitle = 'Temperature'
vp.load('data/images/light.jpg').scale(.00118).pos([.72,-.11,.14])

wavelengths = arange(400, 700, 10)*1e-9
intensities = []
for T in range(3000, 9000, 50):
    I = planck(wavelengths, T)
    coords = list(zip(wavelengths*2e+6, I*0.02, [T*5e-5]*len(I)))
    lineact = vp.line(coords, lw=4, alpha=.5)
    pointColors(lineact, wavelengths*2e+6, cmap='jet')
    vp.show(elevation=.1, azimuth=0.1)

vp.show(interactive=1)
