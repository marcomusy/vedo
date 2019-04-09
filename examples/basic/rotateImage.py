"""
Normal jpg/png images can be loaded,
cropped and rendered as any vtkImageActor
"""
from vtkplotter import Plotter, Text, datadir

vp = Plotter(axes=3, verbose=0)

for i in range(5):
    a = vp.load(datadir+"images/dog.jpg").crop(bottom=0.2) # crop 20%
    a.scale(1 - i / 10.0).alpha(0.8)       # image can be scaled in size
    a.rotateX(20 * i).pos([0, 0, 30 * i])  # (can concatenate methods)

vp.add(Text(__doc__))

vp.show()
