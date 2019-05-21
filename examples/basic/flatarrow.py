"""
FlatArrow example
"""
from vtkplotter import *

for i in range(10):
    s, c = sin(i), cos(i)
    l1 = [[sin(x)+c,     -cos(x)+s,        x] for x in arange(0,3, 0.1)]
    l2 = [[sin(x)+c+0.1, -cos(x)+s + x/15, x] for x in arange(0,3, 0.1)]

    FlatArrow(l1, l2, c=i, tipSize=1, tipWidth=1)

show(collection(), viewup="z", axes=1, bg="w")
