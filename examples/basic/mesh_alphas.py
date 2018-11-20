#Create a map of transparencies which can be passed
# to method pointColors()
#
from vtkplotter import Plotter, arange

vp = Plotter(axes=6) # type 6 marks bounding box corners

act = vp.load('data/beethoven.ply')

# pick y coordinates of vertices and use them as scalars
pscals = act.coordinates()[:,1]

# make a range of transparencies from bottom (0) to top (1)    
alphas = arange(0,1, 1./len(pscals))

act.pointColors(pscals, alpha=alphas, cmap='copper')

vp.show(act)
