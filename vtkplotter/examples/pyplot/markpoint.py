"""Lock an object orientation
to the scene camera"""
from vtkplotter import *

plt = Plotter(bg='bb', axes=1, interactive=0)

sp = Sphere().wireframe()
tx1 = Text("Fixed Text",    sp.points(10), s=0.07, depth=0.1, c="lb")
tx2 = Text("Follower Text", sp.points(144), s=0.07, c="lg")

# a camera must exist prior to calling followCamera()
plt.show(sp)
tx2.followCamera() # a vtkCamera can also be passed as argument

plt.show(sp, tx1, tx2, __doc__, interactive=1)
