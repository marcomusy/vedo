'''
Form a surface by joining two close lines.
'''
from vtkplotter import *


l1 = [ [sin(x),     cos(x),      x/2] for x in arange(0,9, .1)]
l2 = [ [sin(x)+0.2, cos(x)+x/15, x/2] for x in arange(0,9, .1)]

t1 = tube(l1, c='g', r=0.02)
t2 = tube(l2, c='b', r=0.02)

r = ribbon(l1, l2, alpha=.2, res=(200,5), legend='ruled surf').wire(1)

doc = text(__doc__)
show([r,t1,t2, doc], viewup='z', verbose=0)
