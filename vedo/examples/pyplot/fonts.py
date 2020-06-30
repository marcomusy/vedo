from vedo import *
from vedo.settings import fonts

Text2D("List of available fonts", pos='top-center', bg='k', s=1.1)

fonts = ['Arial', 'Courier'] + fonts + ['Times']

for i, f in enumerate(fonts):
    printc('Font: ', f)
    Text2D(f+':  The quick fox jumps over the lazy dog. 1234567890',
           pos=(.01, 1-(i+2)*.055), font=f, s=1.5, c='k')

printc('More fonts available at:')
printc('\thttps://www.1001freefonts.com', underline=1)
printc('\t(use them by setting font=filename.ttf)')

# three points, aka ellipsis, retrieves the list of all created objects
show(..., bg2='cornsilk', axes=False, zoom=1.2, size=(1300,1000))
