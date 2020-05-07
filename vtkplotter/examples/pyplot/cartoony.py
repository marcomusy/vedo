"""Give a cartoony appearance to a 3D mesh"""
from vtkplotter import *

Plotter() # creates a default camera, needed by silhouette()

man = load(datadir+'man.vtk').lighting('off').c('pink').alpha(0.5)
ted = load(datadir+'teddy.vtk').lighting('off').c('sienna')
ted.scale(0.4).rotateZ(-45).pos(-1,-1,-1)

show(
     ted, ted.silhouette(),
     man, man.silhouette().lineWidth(3).color('dr'),
     Text2D(__doc__, pos=5, font="Komiko", s=1.2, bg='dg'),
     bg='wheat', bg2='lb',
     elevation=-80, zoom=1.2,
    )
