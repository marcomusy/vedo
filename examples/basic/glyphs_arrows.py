"""
Draw color arrow glyphs.
"""
from vtkplotter import *
import numpy as np

s1 = Sphere(r=10, res=8).c('white').wireframe()
s2 = Sphere(r=20, res=8).c('white').wireframe().alpha(0.1).pos(0,4,0)

coords1 = s1.coordinates() # get the vertices coords
coords2 = s2.coordinates()

# color can be a colormap which maps arrrow sizes
a1 = Arrows(coords1, coords2, c='coolwarm', alpha=0.4).addScalarBar(c='w')


# get a list of random rgb colors
nrs = np.random.randint(0, 10, len(coords1))
cols = getColor(nrs)

a2 = Arrows(coords1, coords2, c=cols, scale=0.5)

t1 = Text('Color arrows by size\nusing a color map')
t2 = Text('Color arrows by an array\nand scale them')

# draw 2 groups of actors on two renderers
show([[s1, s2, a1, t1], [s1, s2, a2, t2]], N=2)

