import vtkplotter

vp = vtkplotter.Plotter(axes=2)

myted1 = vp.load('data/shapes/teddy.vtk')

myted2 = myted1.clone(mirror='y') # mirror mesh along y-axis
myted2.pos([0,3,0]).color('green')

vp.show([myted1, myted2])

