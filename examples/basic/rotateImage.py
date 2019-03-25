"""
Normal jpg/png images can be 
loaded and rendered as any vtkImageActor
"""
from vtkplotter import Plotter, Text

vp = Plotter(axes=3, verbose=0)

for i in range(5):
    a = vp.load(datadir+"images/dog.jpg")
    a.scale(1 - i / 10.0).alpha(0.8)  # image can be scaled in size
    a.rotateX(20 * i).pos([0, 0, 30 * i])  # (can concatenate methods)

vp.add(Text(__doc__, pos=2))

vp.show()
