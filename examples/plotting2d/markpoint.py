"""Lock an object orientation to the scene camera. 
"""
from vtkplotter import Sphere, Text, Plotter

plt = Plotter(axes=1, interactive=0)

sp = Sphere().wireframe()
tx1 = Text("Fixed point",  sp.getPoint( 10), s=0.07, c="lb")
tx2 = Text("Follow point", sp.getPoint(144), s=0.07, c="lg")

# a camera must exist prior to calling followCamera()
plt.show(sp)
tx2.followCamera() # a vtkCamera can also be passed as argument

plt.show(sp, tx1, tx2, Text(__doc__), interactive=1)
