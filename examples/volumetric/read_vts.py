"""Read structured grid data and show
the associated vector and scalar fields"""
from vedo import *

settings.useDepthPeeling = True

g = load(dataurl+'structgrid.vts')

coords = g.points()

# g.print() gives the list of point and cell data contained in g
vects  = g.pointdata['Momentum']/600 
print('numpy array shapes are:', coords.shape, vects.shape)

# build arrows from starting points to endpoints, with colormap
arrows = Arrows(coords-vects, coords+vects, c='hot_r')

g.cmap('jet', input_array='Density').lineWidth(0.1).alpha(0.3)

show(g, arrows, __doc__, axes=7, viewup='z').close()
