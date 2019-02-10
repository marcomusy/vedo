from vtkplotter import Plotter, Sphere
from vtkplotter.analysis import surfaceIntersection

vp = Plotter()

car = vp.load('data/shapes/porsche.ply', c='gold').alpha(0.1)

s = Sphere(r=4, c='v', alpha=0.1).wire(True)# color is violet

# Intersect car with Sphere, c=black, lw=line width
contour = surfaceIntersection(car, s, lw=4) 

vp.show([car, contour, s], zoom=1.3)
