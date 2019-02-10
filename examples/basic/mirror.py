'''
Mirror a mesh along one of the Cartesian axes.
'''
from vtkplotter import Plotter, Text

vp = Plotter(axes=2)

myted1 = vp.load('data/shapes/teddy.vtk')

myted2 = myted1.clone().mirror('y').pos([0,3,0]).color('green')

vp.show([myted1, myted2, Text(__doc__)], viewup='z')

