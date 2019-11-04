from vtkplotter import Cube, Text, show, collection
from vtkplotter.settings import fonts

Text("List of available fonts:")

Cube().c('white').rotateX(20).rotateZ(20)

for i, f in enumerate(fonts):
    Text(f+':  The quick fox jumps over the lazy dog.',
         pos=(5,i*40+20), font=f, c=i%3)

# three points, aka ellipsis, retrieves the list of all created actors
show(..., axes=False)
