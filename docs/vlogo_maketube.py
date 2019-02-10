'''
logo
'''
from vtkplotter import *

ln = [ [sin(x), cos(x), x/2] for x in arange(0,9, 0.1)]
N = len(ln)

vp = Plotter(verbose=0, axes=0, bg=(57,57,57))
vp.camera.SetPosition( [-0.088, 6.198, 12.757] )
vp.camera.SetFocalPoint( [-0.105, 0.105, 2.209] )
vp.camera.SetViewUp( [-0.294, -0.827, 0.478] )
vp.camera.SetDistance( 12.181 )
vp.camera.SetClippingRange( [6.344, 19.557] )

rads = [ 0.3*(cos(6.*ir/N))**2+0.1 for ir in range(N) ]
cols = [ -i for i in range(N)]
cols = makeBands(cols, 5) # make color bins
t = Tube(ln, r=rads, c=cols, res=24)
vp.show(t)
