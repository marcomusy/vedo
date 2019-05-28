"""Embed a 3D scene 
in a webpage with 
x3dom and vtkplotter"""
from vtkplotter import *

e = load(datadir+'embryo.tif', threshold=True).decimate(0.5)
ec = e.coordinates()
e.pointColors(ec[:,1]) # add dummy colors along y

t = Text(__doc__, pos=[3000., 2000., 4723], s=150, c='w', depth=0.1)
show(t, e)

# This exports the scene and generates 2 files: 
# embryo.x3d and an example embryo.html to inspect in the browser
exportWindow('embryo.x3d')

