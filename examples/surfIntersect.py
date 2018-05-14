import plotter

vp = plotter.vtkPlotter()

# alpha value (opacity) can be put in color string separated by space,/
car = vp.load('data/shapes/alfa147.vtk', c='gold, 0.1') 
s = vp.sphere(r=0.8, c='v/0.1', wire=1) # color is violet with alpha=0.1

c = vp.surfaceIntersection(car, s, c='k', lw=4) # c=black, lw=line width

vp.show([car, c, s])
