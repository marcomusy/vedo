"""Color lines by a scalar"""
from vedo import *

pts1 = [(sin(x/8), cos(x/8), x/5) for x in range(25)]
l1 = Line(pts1).c('black')
l2 = l1.clone().rotateZ(180).shift(1,0,0)

dist = mag(l1.points()-l2.points())  # make up some scalar values

# The trick here is to think that the "body" of a line is a cell
# so we can color cells as we do for any other polygonal mesh:
lines = Lines(l1, l2).lw(4).cmap('Accent', dist, on='cells')

lines.addScalarBar(title='distance') # or e.g.:
# lines.addScalarBar3D(title='distance').scalarbar.rotateX(90).pos(1,1,2)

show(l1,l2, lines, __doc__, axes=1, bg2='lightblue', viewup='z')
