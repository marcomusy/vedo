"""Metrics of quality for
the cells of a triangular mesh.
"""
from vtkplotter import *

mesh = load(datadir+"bunny.obj").scale(100).computeNormals()
mesh.lineWidth(0.1)

# generate an array for mesh quality
arr = mesh.quality(cmap='RdYlBu')

#printHistogram(arr, title='mesh quality', c='w')
hist = histogram(arr, title='mesh quality', bc='w')
hist.scale(.2).pos(-8,4,-9) # make it bigger and place it

# add a scalar bar for the active scalars
mesh.addScalarBar()

# create numeric labels of active scalar on top of cells
labs = mesh.labels(cells=True, precision=3).color('k')

show(mesh,
     labs,
     hist,
     Text2D(__doc__, c='w'),
     bg='bb',
     zoom=2,
     )
