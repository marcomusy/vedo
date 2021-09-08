"""Color lines by a scalar
Click the lines to get their lenghts"""
from vedo import *

settings.allowInteraction=0

pts1 = [(sin(x/8), cos(x/8), x/5) for x in range(25)]
l1 = Line(pts1).c('black')
l2 = l1.clone().rotateZ(180).shift(1,0,0)

dist = mag(l1.points()-l2.points())  # make up some scalar values

# The trick here is to think that the "body" of a line is a cell
# so we can color cells as we do for any other polygonal mesh:
lines = Lines(l1, l2).lw(8).cmap('Accent', dist, on='cells').addScalarBar('length')

def clickfunc(evt):
    if evt.actor:
        idl = evt.actor.closestPoint(evt.picked3d, returnCellId=True)
        print('clicked line', idl, 'length =', precision(dist[idl],3))

plt = Plotter(axes=1, bg2='lightblue')
plt.addCallback('mouse click', clickfunc)
plt.show(l1,l2, lines, __doc__, viewup='z').close()
