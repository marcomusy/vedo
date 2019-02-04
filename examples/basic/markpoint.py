"""
Mark a specific point on a mesh with some text.
"""
from vtkplotter import sphere, point, show, text

sp = sphere().wire(True)

pcoords = sp.point(144)

pt = point(pcoords, r=12, c='white')

tx = text('my fave\npoint', pcoords, s=0.1, 
          c='lightblue', bc='green', followcam=False)

show([sp, pt, tx, text(__doc__)], verbose=0)


