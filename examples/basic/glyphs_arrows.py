"""
Draw color arrow glyphs.
"""
from vedo import *
import numpy as np

s1 = Sphere(r=10, res=8).wireframe().c('white')
s2 = Sphere(r=20, res=8).wireframe().c('white',0.1).pos(0,4,0)

coords1 = s1.points() # get the vertices coords
coords2 = s2.points()

# --- color can be a colormap which maps arrrow sizes
t1 = 'Color arrows by size\nusing a color map'
a1 = Arrows(coords1, coords2, c='coolwarm', alpha=0.4).addScalarBar(c='w')


# --- get a list of random rgb colors
nrs = np.random.randint(0, 10, len(coords1))
cols = getColor(nrs)

t2 = 'Color arrows by an array\nand scale them by half'
a2 = Arrows(coords1, coords2, c=cols)

# draw 2 groups of objects on two renderers
show([(s1, s2, a1, t1), (s1, s2, a2, t2)], N=2, bg='bb', bg2='lb').close()
