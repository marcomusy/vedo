from vedo import *
from vedo.settings import fonts

Text2D("List of available fonts:")

Cube().c('grey').rotateX(20).rotateZ(20)

for i, f in enumerate(fonts):
    printc('Font: ', f)
    Text2D(f+':  The quick fox jumps over the lazy dog. 12345',
           pos=(.01,1-(i+1.5)*.07), font=f, c='k')

# three points, aka ellipsis, retrieves the list of all created objects
show(..., bg='wheat', bg2='white', axes=False, zoom=1.2)
