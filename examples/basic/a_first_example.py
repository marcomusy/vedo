"""
-----------------------------------------------------------------------
A first example: how to draw any mesh.

from vtkplotter import *

c = Cube(side=1.5, alpha=0.5)
s = Sphere()

c.show()        # draw the cube only
(c+s).show()    # Actor+Actor = Assembly, then show it.
show(c,s)       # can show list of [Actor, Volume, filename ...]

vp = Plotter()  # Make a new Plotter() instance and pop up a new window
vp.add(Torus()) # generate another mesh and add it to the Plotter list
vp.show()       # no argument needed
-----------------------------------------------------------------------
"""
print(__doc__)

from vtkplotter import *

c = Cube(side=1.5, alpha=0.5)
s = Sphere()

print("-> c.show()      # press q to continue")
c.show(verbose=0)  # draw the cube only

print("-> (c+s).show()  # Actor+Actor = Assembly, then show it.")
(c + s).show()

print("-> show(c,s)   # can show list of [Actor, Volume, filename ...]")
show(c, s)

print("\n-> # Make a new Plotter() instance and pop up a new window\n-> vp = Plotter()")
vp = Plotter(verbose=0)

print("-> vp.add(Torus()) # generate another mesh and add to the Plotter list")
vp.add(Torus())  # generate another mesh and add to Plotter

print("-> vp.show()       # no argument needed")
vp.show()
