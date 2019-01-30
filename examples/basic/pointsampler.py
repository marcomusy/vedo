"""
This filter functions as follows. First, it regurgitates all input points, 
then samples all lines, plus edges associated with the input polygons and 
triangle strips to produce edge points. Finally, the interiors of polygons 
and triangle strips are subsampled to produce points. 
All of these functions can be enabled or disabled separately. 
Note that this algorithm only approximately generates points the 
specified distance apart. 
"""
print(__doc__)
from vtkplotter import *

s = load('data/shuttle.obj').wire()
t = pointSampler(s).pointSize(4).color('k')
show([s,t])
