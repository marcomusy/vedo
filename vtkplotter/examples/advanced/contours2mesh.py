"""Form a surface mesh
by joining countour lines
"""
import numpy as np
from vtkplotter import *

cs = []
for i in range(-10, 10):
    r = 10 / (i * i + 10)
    c = Circle(r=r).rotateY(i*2).z(i/10).x(i/20)
    c.color('b').alpha(0.1).lineWidth(3).lineColor('k')
    cs.append(c)

# create the mesh by merging the ribbon strips
rbs = []
for i in range(len(cs) - 1):
    rb = Ribbon(cs[i], cs[i+1], closed=True, res=(150,5))
    rbs.append(rb)
mesh = merge(rbs).color('limegreen')

cs.append(Text2D(__doc__))

show([cs, mesh], N=2, axes=1, elevation=-40, bg2='lb')
