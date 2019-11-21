"""
Mirror a mesh along one of the Cartesian axes.

Hover mouse to see original and mirrored.
"""
from vtkplotter import Plotter, Text, datadir

vp = Plotter(axes=2)

myted1 = vp.load(datadir+"teddy.vtk").flag('original')

myted2 = myted1.clone().mirror("y").pos([0, 3, 0]).c("green").flag('mirrored')

vp.show(myted1, myted2, Text(__doc__), viewup="z")
