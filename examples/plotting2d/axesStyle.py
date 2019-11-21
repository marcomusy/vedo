from vtkplotter import *

Earth(r=1).x(3)  # set position x=3

Sphere(r=.3).x(-3).texture('marble2')

printc('press keypad 1-9 to change axes style', box='-', invert=1)
show(..., axes=11, bg='db', bg2='k', verbose=0)
