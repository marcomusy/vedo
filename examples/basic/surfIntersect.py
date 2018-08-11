import vtkplotter

vp = vtkplotter.Plotter()

# alpha value (opacity) can be put in color string separated by space,/
car = vp.load('data/shapes/porsche.ply', c='gold, 0.1') 
s = vp.sphere(r=4, c='v/0.1', wire=1) # color is violet with alpha=0.1

# Intersect car with sphere, c=black, lw=line width
contour = vtkplotter.analysis.surfaceIntersection(car, s, c='k', lw=4) 

vp.show([car, contour, s], zoom=1.3)
