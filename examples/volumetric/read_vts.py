"""Read structured grid data and show
the associated vector and scalar fields
"""
from vedo import load, datadir, Arrows, show

g = load(datadir+'structgrid.vts')#.printInfo()

coords = g.points()
vects  = g.getPointArray('Momentum')/600 # printInfo gives the list
print('numpy array shapes are:', coords.shape, vects.shape)

# build arrows from starting points to endpoints, with colormap
arrows = Arrows(coords-vects, coords+vects, c='hot_r')

g.cmap('jet', input_array='Density').lineWidth(0.1).alpha(0.3)

show(g, arrows, __doc__, axes=7)
