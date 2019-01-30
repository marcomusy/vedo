'''
Text jutification and positioning.
 (center, top-left, top-right, bottom-left, bottom-right)
'''
print(__doc__)
from vtkplotter import show, text, point

txt = 'I like\nto play\nguitar'

tx = text(txt, pos=[1,2,0], s=0.2, c=3, bc=1, depth=0.2, justify='top-left')
to = point([1,2,0], c='r') # mark text origin
ax = point([0,0,0])        # mark axes origin

show([tx,to,ax], axes=8, verbose=0)
