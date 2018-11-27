# Form a surface by joining two lines.
#
from vtkplotter import Plotter, arange, sin, cos

vp = Plotter()

l1 = [ [sin(x),     cos(x),      x/2] for x in arange(0,9, .1)]
l2 = [ [sin(x)+0.2, cos(x)+x/15, x/2] for x in arange(0,9, .1)]

vp.tube(l1, c='g', r=0.02)
vp.tube(l2, c='b', r=0.02)

vp.ribbon(l1, l2, alpha=.2, res=(200,5), legend='ruled surf').wire(1)

vp.show()
