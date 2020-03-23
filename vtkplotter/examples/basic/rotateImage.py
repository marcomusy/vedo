"""Normal jpg/png pictures can be loaded,
cropped, rotated and positioned in 3D
"""
from vtkplotter import Plotter, Text2D, datadir

vp = Plotter(axes=3, verbose=0)

for i in range(5):
    p = vp.load(datadir+"images/dog.jpg") # returns Picture
    p.crop(bottom=0.2)             # crop 20%
    p.scale(1-i/10.0).alpha(0.8)   # picture can be scaled in size
    p.rotateX(20*i).pos(0,0,30*i)  # (can concatenate methods)

vp += Text2D(__doc__)
vp.show()
