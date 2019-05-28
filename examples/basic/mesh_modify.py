"""
Modify mesh vertex positions.
"""
from vtkplotter import *

dsc = Disc().lineWidth(0.1)
coords = dsc.coordinates()
t = Text(__doc__)

for i in range(50):
    coords[:,2] = sin(i/10.*coords[:,0])/5 # move vertices in z
    dsc.setPoints(coords)  # modify mesh
    show(dsc, t, axes=8, resetcam=0, interactive=0)

interactive()