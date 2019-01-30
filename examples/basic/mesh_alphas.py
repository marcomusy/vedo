'''
Create a set of transparencies which can be passed to method pointColors()
'''
from vtkplotter import Plotter, arange, text

vp = Plotter(axes=6) # type 6 = mark bounding box corners

mesh = vp.load('data/beethoven.ply')

# pick y coordinates of vertices and use them as scalars
scals = mesh.coordinates()[:,1]

# define opacities in the range of the scalar, 
# at min(scals) alpha is 0.1, 
# at max(scals) alpha is 0.9:
alphas = [0.1, 0.1, 0.3, 0.4, 0.9]
# or e.g.:
#alphas = arange(0.1, 0.9, 1./len(scals))

mesh.pointColors(scals, alpha=alphas, cmap='copper')
#print(mesh.scalars('pointColors_copper')) # retrieve scalars

vp.add(text(__doc__))
vp.show()
