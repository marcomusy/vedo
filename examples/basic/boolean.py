"""
Example of boolean operations with actors or polydata
"""
print(__doc__)

from vtkplotter import Plotter, booleanOperation, Sphere


# declare the instance of the class
vp = Plotter(shape=(2, 2), interactive=0, axes=3)

# build to sphere actors
s1 = Sphere(pos=[-0.7, 0, 0], c="r", alpha=0.5)
s2 = Sphere(pos=[0.7, 0, 0], c="g", alpha=0.5)

# make 3 different possible operations:
b1 = booleanOperation(s1, "intersect", s2).c('m').legend("intersect")
b2 = booleanOperation(s1, "plus", s2).c("b").wire(True).legend("plus")
b3 = booleanOperation(s1, "minus", s2).legend("minus")

# show the result in 4 different subwindows 0->3
vp.show(s1, s2, at=0)
vp.show(b1, at=1)
vp.show(b2, at=2)
vp.show(b3, at=3)
vp.addScalarBar()  # adds a scalarbar to the last actor
vp.show(interactive=1)
