"""Give a cartoony appearance to a 3D mesh"""
from vedo import *

Plotter() # creates a default camera, needed by silhouette()

man = load(datadir+'man.vtk').lighting('off').c('pink').alpha(0.9)

ted = load(datadir+'teddy.vtk').lighting('off').c('sienna')
ted.scale(0.4).rotateZ(-45).pos(-1,-1,-1)

emb = load(datadir+'embryo.tif').isosurface().extractLargestRegion()
emb.scale(1e-04).rotateZ(-90).pos(0.5,-0.5,-1.4)
emb.lighting('off').c('turquoise')

show(ted, ted.silhouette(),
     emb, emb.silhouette(),
     man, man.silhouette(featureAngle=40).lineWidth(3).color('dr'),
     Text2D(__doc__, pos=5, font="Bongas", s=1.3, bg='dg'),
     bg='wheat', bg2='lb',
     elevation=-80, zoom=1.2,
    )
