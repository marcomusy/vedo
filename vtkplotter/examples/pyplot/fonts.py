from vtkplotter import *
from vtkplotter.settings import fonts

Text2D("List of available fonts:")

Cube().c('white').rotateX(20).rotateZ(20)

for i, f in enumerate(fonts):
    Text2D(f+':  The quick fox jumps over the lazy dog.',
           pos=(5,i*40+20), font=f, c=i)

# three points, aka ellipsis, retrieves the list of all created objects
show(..., bg='wheat', axes=False)
